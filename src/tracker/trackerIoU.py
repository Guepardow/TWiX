import sys
import argparse
from time import time
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from loguru import logger
from scipy.optimize import linear_sum_assignment

sys.path.append('..')
from detection import IDX2COCO
from datasets import init_scene
from structures.tracker import Tracker
from structures.baseDataset import BaseDataset
from association.iou import get_cost_matrix_IoU
from structures.obsCollection import ObsCollection


class TrackerIoU(Tracker):

    def __init__(self, scene: BaseDataset, tracklets: Dict[int, ObsCollection], theta_s: float):

        super().__init__(scene, tracklets)

        # Assign an objectID for objects on the first frame
        self.tracklets[self.scene.first_frame] = self.get_new_identities(self.tracklets[self.scene.first_frame])

        self.theta_s = theta_s                      # minimum similarity to obtain for fusing two consecutive elements
        self.all_tracks = self.get_all_tracks()

    def short_term_association_IoUH(self):

        for frame in self.scene.list_frames[1:]:

            current_obsColl = self.tracklets.get(frame)
            previous_obsColl = self.tracklets.get(frame - self.scene.framestep)

            cost_matrix = get_cost_matrix_IoU(previous_obsColl, current_obsColl, measure='IoU')
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_ind, col_ind):
                if (previous_obsColl[i].classe == current_obsColl[j].classe) and (1-cost_matrix[i, j] > self.theta_s):  # it is a match
                    current_obsColl[j].objectID = previous_obsColl[i].objectID

            # Update tracklets and add new objectIDs if necessary
            self.tracklets[frame] = self.get_new_identities(current_obsColl)

        self.all_tracks = self.get_all_tracks()

def main(opts):

    total_frames, total_time = 0, 0

    # Create folder
    name_param = f"{opts.min_score:0.2f}_{opts.min_area:0.0f}_{opts.theta_s:0.2f}"
    path_to_results_tracker = Path(f"../../results/{opts.dataset}-{opts.subset}/Tracking/TrackerIoU/{opts.detection}/{name_param}")
    path_to_results_tracker.mkdir(parents=True, exist_ok=True)

    # Select the dataset and load the scene
    scene = init_scene(opts.dataset, opts.subset)
    list_scenes = scene.list_scenes if opts.scene is None else [opts.scene]
    logger.info(f"Number of sequences in {opts.dataset}-{opts.subset}: {len(list_scenes)}")

    for scene_name in tqdm(list_scenes, leave=False, desc=f"TrackerIoU on {opts.dataset}-{opts.subset}"):

        # Load scene
        scene.load_scene(scene_name)

        # Load the detections
        filename = Path(f"../../results/{opts.dataset}-{opts.subset}/Detection/{opts.detection}/{scene_name}.txt")
        all_detections = scene.load_detections(filename, dict_classe=IDX2COCO)
        for _, detections in all_detections.items():
            detections.keep_high_score(min_score=opts.min_score)
            detections.keep_big_objects(min_area=opts.min_area)

        # Launch the tracker
        start_timer = time()
        tracker = TrackerIoU(scene=scene, tracklets=all_detections, theta_s=opts.theta_s)
        tracker.short_term_association_IoUH()
        total_time += time()-start_timer
        total_frames += len(tracker.scene.list_frames)

        # Save the results
        tracker.save(path_to_results_tracker, viz=False, benchmark=True, level=tracker.scene.level)

        # Check that all tracks are tracklets (temporal continuity)
        for objectID, obsColl in tracker.all_tracks.items():
            assert obsColl.is_complete(framestep=scene.framestep), f"ObjectID {objectID} is not a tracklet"

    logger.info(f"Elapsed time: {total_time: 0.3f} seconds ({total_frames/total_time:0.2f} Hz)")

    # HOTA Evaluation
    if 'test' not in opts.subset:
        scene.evaluate_performance(split_to_eval=scene.subset,
                                   trackers_folder='../../results',
                                   trackers_to_eval=f'Tracking/TrackerIoU/{opts.detection}',
                                   sub_folder=name_param)

def get_parser():

    parser = argparse.ArgumentParser(description="Multiple Object Tracking - Tracking to get tracklets")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--subset", type=str, required=True, help="Name of the split")
    parser.add_argument("--scene", type=str, default=None, help="Scene")

    # Detection step
    parser.add_argument('--detection', type=str, help="Detector's name")
    parser.add_argument('--min_score', type=float, default=0.50, help="Minimal score of detection")
    parser.add_argument('--min_area', type=int, default=128, help="Minimal area of an instance to be kept")

    # Short term association
    parser.add_argument('--theta_s', type=float, default=0.15, help="Maximum cost to get an association")

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    main(args)
    

# python trackerIoU.py --dataset MOT17 --subset val_half --detection bytetrack_x_mot17
# python trackerIoU.py --dataset DanceTrack --subset val --detection bytetrack_model
# python trackerIoU.py --dataset KITTIMOT --subset val --detection Permatrack
# python trackerIoU.py --dataset MOT20 --subset val_half --detection bytetrack_x_mot20