import sys
import gzip
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm.autonotebook import tqdm

import torch
from torch.utils.data import Dataset

sys.path.append('../..')
from datasets import init_scene
from structures.tracker import Tracker


def get_parser():
    parser = argparse.ArgumentParser(description="Create data for learning association")

    # Dataset
    parser.add_argument('--dataset', type=str, help="Multiple object tracking dataset")
    parser.add_argument('--subset', type=str, help="Subset")

    # Tracklets based on some detections
    parser.add_argument('--detection', type=str, help="Detection method")

    # Methodology to create the batches
    parser.add_argument('--strategy', type=str, default='frame', help="Strategy to move two temporal windows ('frame' or 'tracklet')")
    parser.add_argument('--max_gap', type=float, help="Maximal temporal gap(in seconds)")
    parser.add_argument('--WP', type=str, default='1f', help="Temporal window for the past ('3s' for '3 seconds' ; '1f' for '1 frame')")
    parser.add_argument('--WF', type=str, default='1f', help="Temporal window for the future ('3s' for '3 seconds' ; '1f' for '1 frame')")

    return parser


def get_window_size(W: str, fps: float) -> int:
    """
    Calculate the window size in frames based on the given window size string and frames per second (fps).

    :param W: The window size string. It should end with 's' for seconds or 'f' for frames.
                    For example, '10s' means 10 seconds and '10f' means 10 frames.
    :param fps: The frames per second rate.
    :return: The window size in frames.

    """
    
    if W[-1] == 's':
        return int(fps * float(W[:-1]))  # from seconds to frames
    elif W[-1] == 'f':
        return int(W[:-1])  # frames
    else:
        raise ValueError("Unknown format for the window size")


class DatasetTWiX(Dataset):

    def __init__(self, dataset: str, subset: str, folder_tracklets: str, max_gap: float, strategy: str, WP: str, WF: str, device='cuda'):
        """
        Create a list of batches made of two sets of tracklets

        :param dataset: name of the dataset
        :param subset: name of the subset
        :param folder_tracklets: folder with the tracklets
        :param max_gap: maximal temporal gap between windows (in seconds)
        :param strategy: strategy to create the pairs of tracklets ('frame' or 'tracklet')
        :param WP: temporal window for the past ("3s" for "3 seconds" ; "1f" for "1 frame")
        :param WF: temporal window for the future ("3s" for "3 seconds" ; "1f" for "1 frame")
        """

        self.dataset = dataset
        self.subset = subset
        self.folder_tracklets = folder_tracklets
        self.max_gap = max_gap
        self.strategy = strategy
        self.WP = WP  # str
        self.WF = WF  # str
        self.device = device

        self.list_batches = []
        self.trackers = {}  # it is loaded in the function load()

    def create_batches(self):

        scene = init_scene(self.dataset, self.subset)
        for scene_name in tqdm(scene.list_scenes, desc=f"Creating data on {self.dataset}-{self.subset}"):
            
            scene.load_scene(scene_name)
            scene.load_oracle_infos()

            list_classes = list(scene.dict_COI.values())

            WP = get_window_size(self.WP, scene.fps)  # int
            WF = get_window_size(self.WF, scene.fps)  # int

            # Read the file with the tracks
            filename = Path(f"../../../results/{self.dataset}-{self.subset}/Tracking/{self.folder_tracklets}/{scene_name}.txt")

            tracklets = scene.load_benchmarks(filename)
            tracker = Tracker(scene, tracklets)

            # Get the true identity of each tracklet
            # NB : the ground truth is only used to filter the pairs of tracklets and avoid to have a FP-FP pair
            dict_gt_objectID = tracker.get_gt_tracklets(tracker.scene.gt_tracklets, tracker.scene.ignored_locs, threshold=0.50)
            # dict_gt_objectID is similar to {objectID: int if TP, -1 if ignored, None if FP}

            # Frames where appears each tracklet
            set_frames = {objectID: set([obs.frame for obs in tracker.all_tracks[objectID]]) for objectID in tracker.all_tracks.keys()}

            if self.strategy == 'frame':
                list_pairs_frames = [(f1, f2, classe) for f1 in scene.list_frames for f2 in scene.list_frames for classe in list_classes if 0 <= (f2 - f1 - 1)/scene.fps <= self.max_gap]

            elif self.strategy == 'tracklet':
                # Frame pairs linked by two tracklets such that:
                # - the tracklets belong to the same class
                # - the temporal distance is less than max_gap seconds
                # - the tracklets are not FP-FP
                # - none of the tracklets falls in an ignored region

                list_objectID = sorted(list(tracker.all_tracks.keys()))

                list_pairs_frames = [(tracker.all_tracks[o1][-1].frame, tracker.all_tracks[o2][0].frame, tracker.all_tracks[o1].mode_class()) for o1 in list_objectID for o2 in list_objectID
                                    if (tracker.all_tracks[o1].mode_class() == tracker.all_tracks[o2].mode_class()) and (0 <= (tracker.all_tracks[o2][0].frame - tracker.all_tracks[o1][-1].frame - 1)/scene.fps <= self.max_gap) and 
                                    ((dict_gt_objectID[o1] != -1) and (dict_gt_objectID[o2] != -1)) and ((dict_gt_objectID[o1] is not None) or (dict_gt_objectID[o2] is not None))]
                list_pairs_frames = sorted(list(set(list_pairs_frames)))

            for frameP, frameF, classe in tqdm(list_pairs_frames, desc=f'Creating batches on {scene_name} counting {len(scene.list_frames)} frames', total=len(list_pairs_frames), leave=False):

                # Tracklets of interest : correct temporal window + correct class
                list_frames_past = np.array([frameP - w for w in range(WP) if frameP - w >= tracker.scene.first_frame])
                list_frames_futr = np.array([frameF + w for w in range(WF) if frameF + w <= tracker.scene.last_frame])

                list_objectID_past = sorted(set([obs.objectID for frame in list_frames_past for obs in tracker[frame] if obs.classe == classe]))
                list_objectID_futr = sorted(set([obs.objectID for frame in list_frames_futr for obs in tracker[frame] if obs.classe == classe]))

                # Get the target Y
                Y = self.load_Y(list_objectID_past, list_objectID_futr, dict_gt_objectID, set_frames)
                    
                # Since we use contrastive learning, we need at least two pairs in the target Y, one positive and one negative at least
                if self.is_valid_Y_for_contrastive(Y):
                    self.list_batches.append({'scene': scene_name, 'fps': scene.fps, 'WP': WP, 'WF': WF,
                                            'frameP': frameP, 'frameF': frameF, 'classe': classe,
                                            'list_frames_past': list_frames_past, 'list_frames_futr': list_frames_futr, 
                                            'list_objectID_past': np.array(list_objectID_past), 'list_objectID_futr': np.array(list_objectID_futr),
                                            'Y': Y})
                    
    def is_valid_Y_for_contrastive(self, Y):
        """
        Check if the tensor Y is valid for training: it should have at least one positive pair associated to at least one negative pair.
        """

        # Check each element in the array
        for i in range(Y.shape[0]):  # Iterate through rows
            for j in range(Y.shape[1]):  # Iterate through columns
                if Y[i, j] == 1:  # If the current element is a positive pair
                    # Check the row and column for at least one 0
                    if (np.any(Y[i] == 0) or np.any(Y[:, j] == 0)):
                        return True
        return False

    @staticmethod
    def load_X(tracker, dict_infos):

        list_frames_past, list_frames_futr = dict_infos['list_frames_past'], dict_infos['list_frames_futr']
        list_objectID_past, list_objectID_futr = dict_infos['list_objectID_past'], dict_infos['list_objectID_futr']
        WP = int(dict_infos['WP'])
        WF = int(dict_infos['WF'])

        # Number of tracklets in the past and future
        NP, NF = len(list_objectID_past), len(list_objectID_futr)

        # Tensors from the past tracklets. NaN is used for padding when a tracklet is too short.
        coordsP = float('nan') * np.ones((WP, NP, 4), dtype=np.float32)    # WP x NP x 4
        framesP = float('nan') * np.ones((WP, NP, 1), dtype=np.float32)    # WP x NP x 1

        for idx_obj, objectID in enumerate(list_objectID_past):
            idx_w = 0
            for frame in list_frames_past:
                list_objectID = [obs.objectID for obs in tracker[frame]]
                if objectID in list_objectID:
                    idx = list_objectID.index(objectID)
                    coordsP[idx_w, idx_obj, :] = tracker[frame][idx].locator.coordinates
                    framesP[idx_w, idx_obj, 0] = tracker[frame][idx].frame
                    idx_w += 1

        # Tensors from the future tracklets. NaN is used for padding when a tracklet is too short.
        coordsF = float('nan') * np.ones((WF, NF, 4), dtype=np.float32)  # WF x NF x 4
        framesF = float('nan') * np.ones((WF, NF, 1), dtype=np.float32)  # WF x NF x 1

        for idx_obj, objectID in enumerate(list_objectID_futr):
            idx_w = 0
            for frame in list_frames_futr:
                list_objectID = [obs.objectID for obs in tracker[frame]]
                if objectID in list_objectID:
                    idx = list_objectID.index(objectID)
                    coordsF[idx_w, idx_obj, :] = tracker[frame][idx].locator.coordinates
                    framesF[idx_w, idx_obj, 0] = tracker[frame][idx].frame
                    idx_w += 1

        return coordsP, framesP, coordsF, framesF
    
    def load_Y(self, list_objectID_past, list_objectID_futr, dict_gt_objectID, set_frames):
        """
        Create the tensor Y to predict the association between two sets of tracklets.
        Here, the ground truth is only generated at the tracklet level and not at the detection level.
        """

        dict_gt_objectID_past = {objectID: dict_gt_objectID[objectID] for objectID in list_objectID_past}
        dict_gt_objectID_futr = {objectID: dict_gt_objectID[objectID] for objectID in list_objectID_futr}

        # Matrix to predict
        NP = len(dict_gt_objectID_past)
        NF = len(dict_gt_objectID_futr)

        Y = -1 * np.ones((NP, NF), dtype=int)

        for i, (objectID1, gt_objectID1) in enumerate(dict_gt_objectID_past.items()):

            for j, (objectID2, gt_objectID2) in enumerate(dict_gt_objectID_futr.items()):

                # Use the ground truth at the tracklet level to fill the Y tensor
                if (gt_objectID1 == -1) or (gt_objectID2 == -1):  # one is in an ignored region
                    Y[i, j] = -1
                elif (gt_objectID1 is None) and (gt_objectID2 is None):  # both are false positive
                    Y[i, j] = -1
                else:
                    Y[i, j] = int(gt_objectID1 == gt_objectID2)

                # Use the IoU and overlap between the tracklets to adjust the Y tensor only for the training
                # NB: This helps to reduce the number of -1 in Y and then increase of the number of positive and negative pairs
                #     This is semi-supervised learning : one part of the annotations come from the ground truth (at the tracklet level) 
                #     and the other part from TrackerIoU (unsupervised using pseudo-labeling)
                    
                # same tracklet, for instance a FP-FP pair
                if (Y[i, j] == -1) and (objectID1 == objectID2):  
                    Y[i, j] = 1
                
                # two tracklets with a temporal overlap, even for FP-FP pairs
                overlap = len(set_frames[objectID1].intersection(set_frames[objectID2]))
                if (Y[i, j] == -1) and (overlap != 0):
                    Y[i, j] = 0

        return Y
    
    def __len__(self):
        return len(self.list_batches)

    def __getitem__(self, idx):
        dict_infos = self.list_batches[idx]
        tracker = self.trackers[dict_infos['scene']]
        coordsP, framesP, coordsF, framesF = DatasetTWiX.load_X(tracker, dict_infos)

        return torch.from_numpy(coordsP).to(self.device), torch.from_numpy(framesP).to(self.device), torch.from_numpy(coordsF).to(self.device), \
            torch.from_numpy(framesF).to(self.device), torch.from_numpy(dict_infos['Y']).to(self.device), dict_infos
    
    def save(self, filename):

        # Create the parent directory if it does not exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Save the data using gzip and pickle
        with gzip.open(filename, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):

        if not Path(filename).exists():
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        
        # Load the batches
        with gzip.open(filename, 'rb') as file:
            datasetTWiX = pickle.load(file)
        
        # Load all trackers
        scene = init_scene(datasetTWiX.dataset, datasetTWiX.subset)
        for scene_name in scene.list_scenes:
            scene.load_scene(scene_name)

            # Read the file with the tracks
            filename = Path(f"../../../results/{datasetTWiX.dataset}-{datasetTWiX.subset}/Tracking/{datasetTWiX.folder_tracklets}/{scene_name}.txt")
            tracklets = scene.load_benchmarks(filename)
            datasetTWiX.trackers[scene_name] = Tracker(scene, tracklets)

        return datasetTWiX
    
    def __str__(self):

        text = f"Number of batches: {len(self)}\n"

        ys = [dict_infos['Y'][i, j] for dict_infos in self.list_batches for i, _ in enumerate(dict_infos['list_objectID_past']) for j, _ in enumerate(dict_infos['list_objectID_futr'])]
        counter_y = Counter(ys)
        n_y = len(ys)

        text += f"\tNumber of pairs: {n_y}\n"
        text += f"\tP/N/IGN number: {counter_y[1]} / {counter_y[0]} / {counter_y[-1]}\n"
        text += f"\tP/N/IGN ratio: {counter_y[1]/n_y:0.2%} / {counter_y[0]/n_y:0.2%} / {counter_y[-1]/n_y:0.2%}\n"

        return text

if __name__ == '__main__':

    args = get_parser().parse_args()
    
    # Create a dataset of batches
    data_batches = DatasetTWiX(args.dataset, args.subset, folder_tracklets=f"TrackerIoU/{args.detection}/0.50_128_0.15", 
                               max_gap=args.max_gap, strategy=args.strategy, WP=args.WP, WF=args.WF)
    data_batches.create_batches()

    # Save the batches
    data_batches.save(f"data/{args.dataset}/TrackerIoU/data_{args.subset}_{args.strategy}_{args.max_gap:0.2f}_{args.WP}_{args.WF}.pt")
    print(data_batches)
    