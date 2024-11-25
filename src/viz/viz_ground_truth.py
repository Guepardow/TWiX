import sys
import argparse
from pathlib import Path

sys.path.append('..')
from datasets import init_scene
from structures.tracker import Tracker


def get_parser():
    parser = argparse.ArgumentParser(description="Multiple Object Tracking - Ground truth visualization")

    parser.add_argument('--dataset', type=str, help="Multiple object tracking dataset")
    parser.add_argument('--subset', type=str, help="Subset of the dataset")
    parser.add_argument('--scene', type=str, default=None, help="Scene's name")

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    scene = init_scene(args.dataset, args.subset)
    list_scenes = scene.list_scenes if args.scene is None else [args.scene]

    path_viz = Path(f"../../data/{args.dataset}/ground_truth/{args.subset}")
    path_viz.mkdir(parents=True, exist_ok=True)

    for scene_idx, scenename in enumerate(list_scenes):

        scene.load_scene(scenename)
        scene.load_oracle_infos()

        print(f"Visualizing the ground truth of the dataset {args.dataset}-{args.subset} on scene {scenename} ({1+scene_idx:02d}/{len(list_scenes):02d})")

        # Tracker with the ground truth tracklets
        tracker = Tracker(scene=scene, tracklets=scene.gt_tracklets)
        tracker.visualize_oracle(Path(path_viz) / f"{scenename}.avi")
