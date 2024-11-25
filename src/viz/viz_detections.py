import sys
import argparse
from pathlib import Path

sys.path.append('..')
from detection import IDX2COCO
from datasets import init_scene
from structures.tracker import Tracker


def get_parser():
    parser = argparse.ArgumentParser(description="Multiple Object Tracking - Detection visualization")

    parser.add_argument('--dataset', type=str, help="Multiple object tracking dataset")
    parser.add_argument('--subset', type=str, help="Subset of the dataset")
    parser.add_argument('--scene', type=str, default=None, help="Scene's name")
    parser.add_argument('--detector', type=str, help="Detector's name")

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    path_to_folder = Path(f"../../results/{args.dataset}-{args.subset}/Detection/{args.detector}")

    scene = init_scene(args.dataset, args.subset)
    list_scenes = scene.list_scenes if args.scene is None else [args.scene]

    for scene_idx, scene_name in enumerate(list_scenes):

        scene.load_scene(scene_name)

        tracklets = scene.load_detections(Path(path_to_folder) / f"{scene_name}.txt", dict_classe=IDX2COCO)

        print(f"Visualizing the detections of the dataset {args.dataset}-{args.subset} on scene {scene_name} ({1+scene_idx:02d}/{len(list_scenes):02d})")
        tracker = Tracker(scene=scene, tracklets=tracklets)
        tracker.visualize(Path(path_to_folder) / f"{scene_name}.avi", show_boxes=True)
