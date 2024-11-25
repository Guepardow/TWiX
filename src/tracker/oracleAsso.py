import sys
import argparse
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from loguru import logger

sys.path.append('..')
from detection import IDX2COCO
from datasets import init_scene
from structures.tracker import Tracker
from structures.baseDataset import BaseDataset
from structures.obsCollection import ObsCollection


class OracleAsso(Tracker):
    """
    A class used to design a tracker which has these properties :
    - removes all FP (less than 0.5 IoU with the ground truth)
    - correctly associates detections to the same object
    """
    def __init__(self, scene: BaseDataset, tracklets: Dict[int, ObsCollection], min_IoU: float = 0.5):
        """
        :param scene: Loaded scene with a dataset and a subset
        :param tracklets: Tracklets with all the detections
        :param min_IoU: minimal IoU overlap to consider a TP
        """

        super().__init__(scene, tracklets)
        self.min_IoU = min_IoU

    def oracle_global_associate(self):
        """
        Removes all false positives (based on min_IoU) and assign the ground identity for true positive detections in a single step
        """

        # Get the ground truth tracks for all gt object
        gt_all_tracks = {}
        for f, obsCollection in self.scene.gt_tracklets.items():
            for obs in obsCollection:
                if obs.objectID not in gt_all_tracks:
                    gt_all_tracks[obs.objectID] = ObsCollection()
                gt_all_tracks[obs.objectID].add_observation(obs)

        # Class of each gt_objectID
        dict_gt_classe = {gt_objectID: gt_obsColl.mode_class() for gt_objectID, gt_obsColl in gt_all_tracks.items()}

        # Assign an objectID for all frames
        for frame in self.scene.list_frames:

            dict_gt_objectIDs = self.tracklets.get(frame).get_gt_frame(self.scene.gt_tracklets[frame],
                                                                       self.scene.ignored_locs[frame], threshold=self.min_IoU,
                                                                       img_width=self.scene.width, img_height=self.scene.height)
            # dict_gt_objectIDs : {0: 1, 1:2, 2: None, 3: -1}, None for FP, -1 for ignored area

            # Assign
            idx_to_remove = []
            for idx, gt_objectID in dict_gt_objectIDs.items():
                if (gt_objectID is not None) and (gt_objectID != -1):  # this is a true positive
                    self.tracklets.get(frame)[idx].objectID = gt_objectID
                    self.tracklets.get(frame)[idx].classe = dict_gt_classe.get(gt_objectID)
                    self.tracklets.get(frame)[idx].flag = '_GT_'
                else:
                    idx_to_remove.append(idx)

            # Remove detections that do not match with any ground truth
            for idx in sorted(idx_to_remove, reverse=True):
                self.tracklets[frame].remove_idx(idx)

        self.all_tracks = self.get_all_tracks()

def get_parser():
    parser = argparse.ArgumentParser(description="Multiple Object Tracking - Oracle tracker with perfect association")

    parser.add_argument('--dataset', type=str, help="Multiple object tracking dataset")
    parser.add_argument('--subset', type=str, help="Subset of the dataset")
    parser.add_argument('--scene', type=str, default=None, help="Scene's name")
    parser.add_argument('--detection', type=str, help="Detector's name")

    parser.add_argument('--min_score', type=float, default=0.10, help="Minimal score of detection")
    parser.add_argument('--min_area', type=int, default=0, help="Minimal area of an instance to be kept")
    parser.add_argument('--min_IoU', type=float, default=0.5, help="Minimal IoU to consider a TP")

    return parser


def main(opts):

    # Create folder
    name_param = f"{opts.min_score:0.2f}_{opts.min_area:0.0f}_{opts.min_IoU:0.1f}"
    path_to_results_tracker = Path(f"../../results/{opts.dataset}-{opts.subset}/Tracking/oracleAsso/{opts.detection}/{name_param}")
    path_to_results_tracker.mkdir(parents=True, exist_ok=True)

    # Select the dataset and load the scene
    scene = init_scene(opts.dataset, opts.subset)
    list_scenes = scene.list_scenes if opts.scene is None else [opts.scene]
    logger.info(f"Number of sequences in {opts.dataset}-{opts.subset}: {len(scene.list_scenes)}")

    for scene_name in tqdm(list_scenes):

        # Load scene info and oracle annotations
        scene.load_scene(scene_name)
        scene.load_oracle_infos()

        # Load the detections
        if opts.detection == 'GT':
           all_detections = scene.gt_tracklets
        else:
           all_detections = scene.load_detections(Path(f"../../results/{opts.dataset}-{opts.subset}/Detection/{opts.detection}/{scene_name}.txt"), dict_classe=IDX2COCO)

        # Remove detections that overlap too much / too small / with a low score confidence
        if opts.detection != 'GT':
           for _, detections in all_detections.items():
               detections.keep_high_score(opts.min_score)
               detections.keep_big_objects(opts.min_area)

        # Launch the tracker
        tracker = OracleAsso(scene, all_detections, min_IoU=opts.min_IoU)
        tracker.oracle_global_associate()

        # Save the results
        tracker.save(path_to_results_tracker, viz=False, benchmark=True, level=tracker.scene.level)

    if 'test' not in opts.subset:
        scene.evaluate_performance(split_to_eval=scene.subset,
                                   trackers_folder='../../results',
                                   trackers_to_eval=f'Tracking/oracleAsso/{opts.detection}',
                                   sub_folder=name_param)


if __name__ == '__main__':

    args = get_parser().parse_args()
    main(args)

# Perfect association on perfect detections : HOTA expected to be 100.0
# python oracleAsso.py --dataset MOT17 --subset val_half --detection GT --min_score 0.0 --min_area 0
# python oracleAsso.py --dataset DanceTrack --subset val --detection GT --min_score 0.0 --min_area 0
# python oracleAsso.py --dataset KITTIMOT --subset val --detection GT --min_score 0.0 --min_area 0
# python oracleAsso.py --dataset MOT20 --subset val_half --detection GT --min_score 0.0 --min_area 0

# python oracleAsso.py --dataset MOT17 --subset val_half --detection bytetrack_x_mot17 --min_score 0.5 --min_area 128
# python oracleAsso.py --dataset DanceTrack --subset val --detection bytetrack_model --min_score 0.5 --min_area 128
# python oracleAsso.py --dataset KITTIMOT --subset val --detection Permatrack --min_score 0.5 --min_area 128
# python oracleAsso.py --dataset MOT20 --subset val_half --detection bytetrack_x_mot20 --min_score 0.5 --min_area 128