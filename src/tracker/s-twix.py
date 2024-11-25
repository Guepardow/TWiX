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
    parser = argparse.ArgumentParser(description="Multiple Object Tracking - S-TWiX")

    # Dataset and subset
    parser.add_argument('--dataset', type=str, default='DanceTrack', help="Multiple object tracking dataset")
    parser.add_argument('--subset', type=str, default='val', help="subset of the dataset")
    parser.add_argument('--scene', type=str, default=None, help="Name of a scene")

    # Detection
    parser.add_argument('--detection', type=str, default='bytetrack_model', help="Detector's name")
    parser.add_argument('--min_score', type=float, default=0.50, help="Minimal confidence score of detections")
    parser.add_argument('--min_area', type=float, default=128, help="Minimal area of detections")

    # Single matching step
    parser.add_argument('--folder_twix', type=str, default='rsta', choices=['rsta', 'rlta'], help="Folder of TWiX")
    parser.add_argument('--method_twix', type=str, help="TWiX config for the match")
    parser.add_argument('--theta', type=float, help="Maximum cost to get a matching")

    # Update tracks and states
    parser.add_argument('--max_age', type=float, help="Maximum age (in seconds) of an unmatched track")
    parser.add_argument('--min_score_new', type=float, help="Minimal confidence score on detections for creating a new tracklet")

    return parser


class Single_TWiX:  
    """
    Tracker Single-TWiX (S-TWiX) based on a single matching pipeline
    """
    def __init__(self, scene, detection, min_score: float, min_area: float, folder_twix: str, method_twix: str, theta: float, 
                 max_age: float, min_score_new: float):
        """
        :param scene: Scene object
        :param detection: Name of the detector
        :param min_score: Minimal confidence score of detections
        :param min_area: Minimal area of detections

        :param folder_twix: folder containing the weights
        :param method_twix: TWiX config folder for the matching algorithm
        :param theta: Minimal similarity to get a matching

        :param max_age: Maximum age (in seconds) of an unmatched track before being killed
        :param min_score_new: Minimal confidence score on detections for creating a new tracklet
        """

        # Properties 
        self.scene = scene

        self.detection = detection
        self.min_score = min_score
        self.min_area = min_area

        self.folder_twix = folder_twix
        self.method_twix = method_twix
        self.theta = theta

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

        # Load weights
        config_file = Path(f"../association/twix/{self.folder_twix}/{self.scene.dataset_name}/{self.method_twix}/args.json")
        with open(config_file, 'r') as file:
            d = yaml.safe_load(file)

        # Load model
        self.model_twix = TWiX(d_model=d['d_model'], nhead=d['nhead'], dim_feedforward=d['dim_feedforward'], num_layers=d['num_layers'], dropout=d['dropout'], inter_pair=d['inter_pair'])
        self.WP = get_window_size(d['WP'], self.scene.fps)

        # Load weights
        path_weights_model = Path(f"../association/twix/{self.folder_twix}/{self.scene.dataset_name}/{self.method_twix}/best_weights.pth.tar")
        self.model_twix.load_state_dict(torch.load(path_weights_model, weights_only=False)['state_dict'])
        self.model_twix.to(DEVICE)
        self.model_twix.eval()

    def load_tensors(self, dict_tracks: Dict[int, Track], detections: List[Observation], WP):

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

    # Select the dataset and load the scene
    scene = init_scene(opts.dataset, opts.subset)
    list_scenes = scene.list_scenes if args.scene is None else [args.scene]
    logger.info(f"Number of sequences in {opts.dataset}-{opts.subset}: {len(list_scenes)}")

    # Create folder
    name_hyper = f'{opts.min_score:0.2f}_{opts.min_area:0.0f}_{opts.method_twix}_{opts.theta:0.2f}_{opts.max_age:0.2f}_{opts.min_score_new:0.2f}'
    path_folder = Path(f"../../results/{scene.dataset_name}-{scene.subset}/Tracking/S-TWiX/{opts.detection}/{name_hyper}")
    path_folder.mkdir(parents=True, exist_ok=True)
    
    total_frames, total_time = 0, 0

    for scene_name in tqdm(list_scenes, desc=f"Tracking on {opts.dataset}-{opts.subset}", leave=False):

        # Load scene
        scene.load_scene(scene_name)

        # Initialize the tracker
        tracker = Single_TWiX(scene=scene, detection=opts.detection,
                               min_score=opts.min_score, min_area=opts.min_area, 
                               folder_twix=opts.folder_twix,
                               method_twix=opts.method_twix, theta=opts.theta, 
                               max_age=opts.max_age, min_score_new=opts.min_score_new)

        start_timer = time()
        for frame in tqdm(tracker.scene.list_frames[1:], desc=f"S-TWiX tracker on {scene_name}", leave=False):

            # Single association matching
            tracker.matching(dict_of_tracks=tracker.alive_tracks, 
                            obsColl=[obs for obs in tracker.all_detections.get(frame) if obs.score >= tracker.min_score], 
                            frame=frame, theta=tracker.theta, model=tracker.model_twix, WP=tracker.WP)
 
            tracker.kill_and_update_tracks()
            tracker.create_new_tracks(frame)
            tracker.update_states()

        total_time += time()-start_timer
        total_frames += len(tracker.all_detections)
        tracker.save(folder=path_folder)

    logger.info(f"Elapsed time: {total_time: 0.3f} seconds ({total_frames/total_time:0.2f} Hz)")

    # Evaluate HOTA
    if 'test' not in opts.subset:
        scene.evaluate_performance(split_to_eval=scene.subset,
                                   trackers_folder='../../results',
                                   trackers_to_eval=f'Tracking/S-TWiX/{opts.detection}',
                                   sub_folder=name_hyper)
        

if __name__ == '__main__':

    args = get_parser().parse_args()
    main(args)