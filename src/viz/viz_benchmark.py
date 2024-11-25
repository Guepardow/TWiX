import sys
import argparse
from pathlib import Path

sys.path.append('..')
from datasets import init_scene
from structures.tracker import Tracker


def get_parser():
    parser = argparse.ArgumentParser(description="Multiple Object Tracking - Tracker visualization")

    parser.add_argument('--dataset', type=str, help="Multiple object tracking dataset")
    parser.add_argument('--subset', type=str, help="Subset of the dataset")
    parser.add_argument('--scene', type=str, default=None, help="Scene's name")
    parser.add_argument('--folder', type=str, help="Path to the file")

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    path_viz = Path(f"../../results/{args.dataset}-{args.subset}/Tracking/{args.folder}")

    scene = init_scene(args.dataset, args.subset)
    list_scenes = scene.list_scenes if args.scene is None else [args.scene]

    for scene_idx, scene_name in enumerate(list_scenes):

        scene.load_scene(scene_name)

        tracklets = scene.load_benchmarks(path_viz / f"{scene_name}.txt")

        print(f"Visualizing the benchmarks of the dataset {args.dataset}-{args.subset} on scene {scene_name} ({1+scene_idx:02d}/{len(list_scenes):02d})")
        tracker = Tracker(scene=scene, tracklets=tracklets)
        tracker.visualize(path_viz / f"{scene_name}.avi", show_boxes=scene.level == 'box')
