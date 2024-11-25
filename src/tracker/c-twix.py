import sys
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from pathlib import Path
from loguru import logger
from typing import List, Dict
from scipy.optimize import linear_sum_assignment

sys.path.append('..')
from detection import IDX2COCO
from datasets import init_scene
from structures.track import Track
from association.twix.twix import TWiX
from structures.observation import Observation
from association.twix.data import get_window_size
from structures.obsCollection import ObsCollection


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    

def get_parser():
    parser = argparse.ArgumentParser(description="Multiple Object Tracking - C-TWiX")

    # Dataset and subset
    parser.add_argument('--dataset', type=str, default='DanceTrack', help="Multiple object tracking dataset")
    parser.add_argument('--subset', type=str, default='val', help="subset of the dataset")
    parser.add_argument('--scene', type=str, default=None, help="Name of a scene")

    # Detection
    parser.add_argument('--detection', type=str, default='bytetrack_model', help="Detector's name")
    parser.add_argument('--min_score', type=float, default=0.50, help="Minimal confidence score of detections")
    parser.add_argument('--min_area', type=float, default=128, help="Minimal area of detections")

    # First matching at STA
    parser.add_argument('--method_twix_1', type=str, help="TWiX config for the first match")
    parser.add_argument('--theta_1', type=float, help="Maximum cost to get a matching")

    # Second matching at LTA
    parser.add_argument('--method_twix_2', type=str, help="TWiX config for the second match")
    parser.add_argument('--theta_2', type=float, help="Maximum cost to get a matching")

    # Update tracks and states
    parser.add_argument('--max_age', type=float, help="Maximum age (in seconds) of an unmatched track")
    parser.add_argument('--min_score_new', type=float, help="Minimal confidence score on detections for creating a new tracklet")

    return parser


class Cascade_TWiX:  
    """
    Tracker Cascade-TWiX (C-TWiX) based on the pipeline of Cascade Buffered IoU (C-BIoU)
    """
    def __init__(self, scene, detection, min_score: float, min_area: float, method_twix_1: str, theta_1: float, 
                 method_twix_2: str, theta_2: float, max_age: float, min_score_new: float):
        """
        :param scene: Scene object
        :param detection: Name of the detector
        :param min_score: Minimal confidence score of detections
        :param min_area: Minimal area of detections

        :param method_twix_1: TWiX config folder for the first match
        :param theta_1: Minimal similarity to get a matching

        :param method_twix_2: TWiX config folder for the second match
        :param theta_2: Minimal similarity to get a matching

        :param max_age: Maximum age (in seconds) of an unmatched track before being killed
        :param min_score_new: Minimal confidence score on detections for creating a new tracklet
        """

        # Properties 
        self.scene = scene

        self.detection = detection
        self.min_score = min_score
        self.min_area = min_area

        self.method_twix_1 = method_twix_1  # For the first matching
        self.theta_1 = theta_1

        self.method_twix_2 = method_twix_2  # For the second matching
        self.theta_2 = theta_2

        self.max_age = max_age
        self.min_score_new = min_score_new

        # Structures for online tracking
        self.alive_tracks = {}  # {objectID:  Track}
        self.unmatched_tracks = {}  # {objectID:  Track} as a subset of self.alive_tracks
        self.unmatched_dets = []    # a list of observations

        # Memory of the tracker : stores all the history of the tracker under the format {objectID: Track}
        # It is updated when a track is too old (aka dead) after each frame iteration and at the end of the tracker (purge)
        self.memory = {} 

        # Load the detections on every frame of the scene
        self.load_detections()

        # Total number of tracked objects
        all_objectIDs = [obs.objectID for obsCollection in list(self.all_detections.values()) for obs in obsCollection]
        self.n_objectIDs = max(all_objectIDs) if all_objectIDs else 0

        # Assign an objectID for objects at the first frame
        self.initialize_first_frame()

        # Load weights of TWiX
        self.load_weights()

    def initialize_first_frame(self):

        # Gives an objectID to all detections at the first frame
        for obs in self.all_detections[self.scene.first_frame]:
            new_objectID = self.n_objectIDs+1
            self.alive_tracks[new_objectID] = Track(objectID=new_objectID, classe=obs.classe, locator=obs.locator, frame=self.scene.first_frame, score=obs.score)
            self.n_objectIDs += 1  # increments the number of total tracked objects

    def load_detections(self):

        if self.detection == 'GT':
            self.scene.load_oracle_infos()
            all_detections = self.scene.gt_tracklets

            for f, obsColl in all_detections.items():
                for obs in obsColl:
                    obs.objectID = 0

        else:
            all_detections = self.scene.load_detections(Path(f"../../results/{self.scene.dataset_name}-{self.scene.subset}/Detection/{self.detection}/{self.scene.scene_name}.txt"), dict_classe=IDX2COCO)

            for _, detections in all_detections.items():
                detections.keep_high_score(min_score=self.min_score)
                detections.keep_big_objects(min_area=self.min_area)
                detections.keep_class(classes_to_keep=self.scene.dict_COI.values())

        self.all_detections = all_detections

    def load_weights(self):

        # Load config for STA
        config_file = Path(f"../association/twix/rsta/{self.scene.dataset_name}/{self.method_twix_1}/args.json")
        with open(config_file, 'r') as file:
            d = yaml.safe_load(file)
        self.SWP = get_window_size(d['WP'], self.scene.fps)

        # Load model
        self.model_twix_sta = TWiX(d_model=d['d_model'], nhead=d['nhead'], dim_feedforward=d['dim_feedforward'], num_layers=d['num_layers'], dropout=d['dropout'], inter_pair=d['inter_pair'], device=DEVICE)

        # Load weights
        path_weights_model = Path(f"../association/twix/rsta/{self.scene.dataset_name}/{self.method_twix_1}/best_weights.pth.tar")
        self.model_twix_sta.load_state_dict(torch.load(path_weights_model, weights_only=False, map_location=DEVICE)['state_dict'])
        self.model_twix_sta.to(DEVICE)
        self.model_twix_sta.eval()

        # Load config for LTA
        config_file = Path(f"../association/twix/rlta/{self.scene.dataset_name}/{self.method_twix_2}/args.json")
        with open(config_file, 'r') as file:
            d = yaml.safe_load(file)
        self.LWP = get_window_size(d['WP'], self.scene.fps)

        # Load model
        self.model_twix_lta = TWiX(d_model=d['d_model'], nhead=d['nhead'], dim_feedforward=d['dim_feedforward'], num_layers=d['num_layers'], dropout=d['dropout'], inter_pair=d['inter_pair'], device=DEVICE)

        # Load weights
        path_weights_model = Path(f"../association/twix/rlta/{self.scene.dataset_name}/{self.method_twix_2}/best_weights.pth.tar")
        self.model_twix_lta.load_state_dict(torch.load(path_weights_model, weights_only=False, map_location=DEVICE)['state_dict'])
        self.model_twix_lta.to(DEVICE)
        self.model_twix_lta.eval()

    def load_tensors(self, dict_tracks: Dict[int, Track], detections: List[Observation], WP: int):

        list_objectID_past = dict_tracks.keys()
        NP = len(list_objectID_past)
        NF = len(detections)

        if len(dict_tracks) == 0:

            coordsP = torch.zeros((WP, 0, 4)).to(DEVICE)
            framesP = torch.zeros((WP, 0, 1)).to(DEVICE)

        else:

            # Tensors from the past tracklets. NaN is used for padding when a tracklet is too short.
            coordsP = float('nan') * np.ones((WP, NP, 4), dtype=np.float32)    # WP x NP x 4
            framesP = float('nan') * np.ones((WP, NP, 1), dtype=np.float32)    # WP x NP x 1

            for idx_obj, objectID in enumerate(list_objectID_past):
                idx_w = 0
                for df in range(min(WP, len(dict_tracks[objectID].coordinates))):
                    coordsP[idx_w, idx_obj, :] = dict_tracks[objectID].coordinates[-df-1]
                    framesP[idx_w, idx_obj, 0] = dict_tracks[objectID].frames[-df-1]
                    idx_w += 1

            coordsP = torch.from_numpy(coordsP).to(DEVICE)   # WP x NP x 4
            framesP = torch.from_numpy(framesP).to(DEVICE)   # WP x NP x 1

        if len(detections) == 0:

            coordsF = torch.zeros((1, 0, 4)).to(DEVICE)
            framesF = torch.zeros((1, 0, 1)).to(DEVICE)

        else:

            # Tensors from the future tracklets. NaN is used for padding when a tracklet is too short.
            coordsF = np.array([obs.locator.coordinates for obs in detections])         # NF x 4
            framesF = np.array([obs.frame for obs in detections])                       # NF

            coordsF = torch.from_numpy(coordsF).unsqueeze(0).to(DEVICE)                   # 1 x NF x 4
            framesF = torch.from_numpy(framesF).unsqueeze(0).unsqueeze(2).to(DEVICE)      # 1 x NF x 1

        assert coordsP.shape == (WP, NP, 4)
        assert framesP.shape == (WP, NP, 1)

        assert coordsF.shape == (1, NF, 4)
        assert framesF.shape == (1, NF, 1)

        return coordsP, framesP, coordsF, framesF

    def matching(self, dict_of_tracks: Dict[int, Track], obsColl: ObsCollection, frame: int, theta: float, model, WP: int):

        self.unmatched_tracks = {}
        self.unmatched_dets = []

        for classe in self.scene.dict_COI.values():

            dict_of_tracks_classe = {objectID: track for objectID, track in dict_of_tracks.items() if track.classe == classe}
            objectIDs = list(dict_of_tracks_classe.keys())

            obsColl_classe = [obs for obs in obsColl if obs.classe == classe]

            if not (objectIDs and obsColl_classe):
                cost_matrix = np.empty((0, 0))

            else:
                with torch.no_grad():
                    with torch.autocast(device_type=DEVICE):
                        coordsP, framesP, coordsF, framesF = self.load_tensors(dict_of_tracks_classe, obsColl_classe, WP=WP)
                        affinities = model(coordsP, framesP, coordsF, framesF, fps=torch.tensor(self.scene.fps))
                        affinities = affinities.detach().cpu().numpy() # between -1 and 1
                        cost_matrix = 1 - affinities
            
            # Match tracks with detections using the Hungarian algorithm
            row_ind_all, col_ind_all = linear_sum_assignment(cost_matrix)

            # Filter assignments that are above a certain level of affinity
            row_ind = [i for i, j in zip(row_ind_all, col_ind_all) if 1-cost_matrix[i, j] > theta]
            col_ind = [j for i, j in zip(row_ind_all, col_ind_all) if 1-cost_matrix[i, j] > theta]
            assert len(row_ind) == len(col_ind)

            # Update matched tracks by concatenating the coordinates      
            for i, j in zip(row_ind, col_ind):
                objectID = objectIDs[i]
                self.alive_tracks[objectID].add_observation(locator=obsColl_classe[j].locator, frame=frame, score=obsColl_classe[j].score)

            for i in range(len(dict_of_tracks_classe)):
                if i not in row_ind:
                    self.unmatched_tracks[objectIDs[i]] = dict_of_tracks_classe[objectIDs[i]]

            for j in range(len(obsColl_classe)):
                if j not in col_ind:
                    self.unmatched_dets = self.unmatched_dets + [obsColl_classe[j]]

    def create_new_tracks(self, frame: int):

        for obs in self.unmatched_dets:

            if obs.score > self.min_score_new:
                new_objectID = self.n_objectIDs+1
                self.alive_tracks[new_objectID] = Track(objectID=new_objectID, classe=obs.classe, locator=obs.locator, frame=frame, score=obs.score)
                self.n_objectIDs += 1  # increments the number of total tracked objects
        
    def kill_and_update_tracks(self):

        objectID_to_kill = []  # need a list to avoid error 'dictionary changed size during iteration'
        for objectID, track in self.unmatched_tracks.items():
            if track.age > self.max_age:
                objectID_to_kill.append(objectID)
                #logger.success(f"Kill the track {objectID} of age {dico['age']}")
            else:
                self.alive_tracks[objectID] = self.unmatched_tracks[objectID]
                self.alive_tracks[objectID].age += 1/self.scene.fps  # age in seconds

        for objectID in objectID_to_kill:
            self.memory[objectID] = self.unmatched_tracks[objectID]
            self.unmatched_tracks.pop(objectID)
            self.alive_tracks.pop(objectID)

    def update_states(self):

        for objectID, track in self.alive_tracks.items():
            self.alive_tracks[objectID].state = track.coordinates[-1]

    def save(self, folder: str):

        # Purge the memory
        for objectID, track in self.alive_tracks.items():
            self.memory[objectID] = track

        self.scene.save_benchmark_from_Track(self.memory, folder=folder)

def main(opts):

    total_frames, total_time = 0, 0

    # Select the dataset and load the scene
    scene = init_scene(opts.dataset, opts.subset)
    list_scenes = scene.list_scenes if args.scene is None else [args.scene]
    logger.info(f"Number of sequences in {opts.dataset}-{opts.subset}: {len(list_scenes)}")

    for scene_name in tqdm(list_scenes, desc=f"Tracking on {opts.dataset}-{opts.subset}"):

        # Load scene info
        scene.load_scene(scene_name)

        # Initialize the tracker
        tracker = Cascade_TWiX(scene=scene, detection=opts.detection,
                               min_score=opts.min_score, min_area=opts.min_area, 
                               method_twix_1=opts.method_twix_1, theta_1=opts.theta_1,
                               method_twix_2=opts.method_twix_2, theta_2=opts.theta_2,
                               max_age=opts.max_age, min_score_new=opts.min_score_new)

        start_timer = time()
        for frame in tqdm(tracker.scene.list_frames[1:], desc=f"C-TWiX tracker on {scene_name}", leave=False):

            # First association matching
            tracker.matching(dict_of_tracks=tracker.alive_tracks, 
                            obsColl=[obs for obs in tracker.all_detections.get(frame) if obs.score >= tracker.min_score], 
                            frame=frame, theta=tracker.theta_1, model=tracker.model_twix_sta, WP=tracker.SWP)

            # Second association matching
            tracker.matching(dict_of_tracks=tracker.unmatched_tracks, 
                            obsColl=tracker.unmatched_dets, 
                            frame=frame, theta=tracker.theta_2, model=tracker.model_twix_lta, WP=tracker.LWP)
 
            tracker.kill_and_update_tracks()
            tracker.create_new_tracks(frame)
            tracker.update_states()

        total_time += time()-start_timer
        total_frames += len(tracker.scene.list_frames)

        # Save the results
        name_hyper = f'{opts.min_score:0.2f}_{opts.min_area:0.0f}_{opts.method_twix_1}_{opts.theta_1:0.2f}_{opts.method_twix_2}_{opts.theta_2:0.2f}_{opts.max_age:0.2f}_{opts.min_score_new:0.2f}'
        folder = Path(f"../../results/{opts.dataset}-{opts.subset}/Tracking/C-TWiX/{opts.detection}/{name_hyper}")
        folder.mkdir(parents=True, exist_ok=True)
        tracker.save(folder=folder)

    logger.info(f"Elapsed time: {total_time: 0.3f} seconds ({total_frames/total_time:0.2f} Hz)")

    # Evaluate HOTA
    if 'test' not in opts.subset:
        scene.evaluate_performance(split_to_eval=scene.subset,
                                   trackers_folder='../../results',
                                   trackers_to_eval=f'Tracking/C-TWiX/{opts.detection}',
                                   sub_folder=f'{opts.min_score:0.2f}_{opts.min_area:0.0f}_{opts.method_twix_1}_{opts.theta_1:0.2f}_{opts.method_twix_2}_{opts.theta_2:0.2f}_{opts.max_age:0.2f}_{opts.min_score_new:0.2f}')


if __name__ == '__main__':

    args = get_parser().parse_args()
    main(args)