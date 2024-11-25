import sys
import argparse
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from loguru import logger
from collections import defaultdict

sys.path.append('..')
from datasets import init_scene
from structures.tracker import Tracker
from structures.baseDataset import BaseDataset
from structures.obsCollection import ObsCollection


class OracleLTA(Tracker):
    """
    A class used to design a tracker which has these properties :
    - use the outputs of a first tracker (usually one that outputs tracklets)
    - correctly associates tracklets
    """
    def __init__(self, scene: BaseDataset, tracklets: Dict[int, ObsCollection], min_IoU: float = 0.5):
        """
        :param scene: Loaded scene with a dataset and a subset
        :param tracklets: Tracklets with all the detections
        :param min_IoU: minimal IoU overlap to consider a TP
        """

        super().__init__(scene, tracklets)
        self.min_IoU = min_IoU

        # Get the true identity of each tracklet
        self.dict_gt_objectID = self.get_gt_tracklets(self.scene.gt_tracklets, self.scene.ignored_locs, threshold=self.min_IoU)
        # {1: 1, 2: None, 3: -1}, None = FP, -1 = ignored

    def oracle_long_term_associate(self):
        """
        Perfectly associate tracklets between them
        """

        # Dictionnary to get gt_objectID: [objectID1, objectID2, ...]
        dict_gt_objectID_list_objectIDs = defaultdict(list)
        for objectID, gt_objectID in self.dict_gt_objectID.items():
            if (gt_objectID is not None) and (gt_objectID != -1):
                dict_gt_objectID_list_objectIDs[gt_objectID].append(objectID)

        # Dictionnaire to get the predicted objectID : gt_objectID: min(objectIDs)
        dict_objectID_objectID = defaultdict(int)
        for objectID, gt_objectID in self.dict_gt_objectID.items():
            if (gt_objectID is not None) and (gt_objectID != -1):
                dict_objectID_objectID[objectID] = min(dict_gt_objectID_list_objectIDs[gt_objectID])

        # Assign an objectID for all frames
        for frame, obsColl in self.tracklets.items():

            # Assign
            for idx, obs in enumerate(obsColl):
                if obs.objectID in dict_objectID_objectID:
                    obs.objectID = dict_objectID_objectID[obs.objectID]

        # In case two observations of the same objectID in the same frame
        self.merge_same_objectIDs()

        self.all_tracks = self.get_all_tracks()
        

def get_parser():
    parser = argparse.ArgumentParser(description="Multiple Object Tracking - Oracle tracker with perfect association at LTA")

    parser.add_argument('--dataset', type=str, help="Multiple object tracking dataset")
    parser.add_argument('--subset', type=str, help="Subset of the dataset")
    parser.add_argument('--scene', type=str, default=None, help="Scene's name")
    parser.add_argument('--folder_tracklets', type=str, help="Path to the tracklets folder")

    parser.add_argument('--min_IoU', type=float, default=0.5, help="Minimal IoU to consider a TP")

    return parser


def main(opts):

    path_to_results_tracker = Path(f"../../results/{opts.dataset}-{opts.subset}/Tracking/oracleLTA/{opts.folder_tracklets}/{opts.min_IoU:0.1f}")
    path_to_results_tracker.mkdir(parents=True, exist_ok=True)

    # Select the dataset and load the scene
    scene = init_scene(opts.dataset, opts.subset)
    list_scenes = scene.list_scenes if opts.scene is None else [opts.scene]
    logger.info(f"Number of sequences in {opts.dataset}-{opts.subset}: {len(scene.list_scenes)}")

    for scene_name in tqdm(list_scenes):

        # Load scene info and oracle annotations
        scene.load_scene(scene_name)
        scene.load_oracle_infos()

        # Load the tracklets from the first tracker
        filename = Path(f"../../results/{opts.dataset}-{opts.subset}/Tracking/{opts.folder_tracklets}/{scene_name}.txt")
        tracklets = scene.load_benchmarks(filename)

        # Launch the tracker on all frames
        tracker = OracleLTA(scene, tracklets, min_IoU=opts.min_IoU)
        tracker.oracle_long_term_associate()

        # Save the results
        tracker.save(path_to_results_tracker, viz=False, benchmark=True, level=tracker.scene.level)

    # HOTA Evaluation
    if 'test' not in opts.subset:
        scene.evaluate_performance(split_to_eval=scene.subset,
                                   trackers_folder='../../results',
                                   trackers_to_eval=f'Tracking/oracleLTA/{opts.folder_tracklets}',
                                   sub_folder=f"{opts.min_IoU:0.1f}")
        

if __name__ == '__main__':

    args = get_parser().parse_args()
    main(args)

# python oracleLTA.py --dataset MOT17 --subset val_half --folder_tracklets TrackerIoU/bytetrack_x_mot17/0.50_128_0.15
# python oracleLTA.py --dataset DanceTrack --subset val --folder_tracklets TrackerIoU/bytetrack_model/0.50_128_0.15
# python oracleLTA.py --dataset KITTIMOT --subset val --folder_tracklets TrackerIoU/Permatrack/0.50_128_0.15
# python oracleLTA.py --dataset MOT20 --subset val_half --folder_tracklets TrackerIoU/bytetrack_x_mot20/0.50_128_0.15
