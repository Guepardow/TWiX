import cv2
import subprocess
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
from loguru import logger

from structures.track import Track
from structures.baseDataset import BaseDataset
from structures.observation import Observation
from structures.locator import Point, Box, Mask
from structures.obsCollection import ObsCollection


PATH_CURRENT = Path(__file__).resolve().parent
PATH_DATA = PATH_CURRENT.parent.parent / 'data'
PATH_BASE = PATH_CURRENT.parent.parent


class DanceTrack(BaseDataset):
    def __init__(self, subset: str):

        self.list_subsets = ['test', 'val', 'train']
        assert subset in self.list_subsets, f"subset {subset} is not in {self.list_subsets}"

        super().__init__('DanceTrack', subset)

        self.dict_COI = {1: 'pedestrian'}
        self.framestep = 1
        self.level = 'box'
        self.list_scenes = sorted([path.name for path in Path(f"{PATH_DATA}/DanceTrack/{subset}").iterdir()])

    def load_scene(self, scene_name):

        if scene_name not in self.list_scenes:
            raise KeyError(f"The scene {scene_name} is not a valid scene name! Acceptable scene names are: \n{self.list_scenes}")
        self.scene_name = scene_name

        self.fps = 20
        self.height, self.width, _ = cv2.imread(self.get_path_image(1)).shape

        self.list_frames = sorted([int(frame.name[:-4]) for frame in Path(f"{PATH_DATA}/DanceTrack/{self.subset}/{scene_name}/img1").iterdir()])

    def load_oracle_infos(self):

        # Open the file with bounding boxes
        df_gt = pd.read_csv(Path(f"{PATH_DATA}/DanceTrack/{self.subset}/{self.scene_name}/gt/gt.txt"),
                            names=['frames', 'objectIDs', 'xmin', 'ymin', 'width', 'height', 'x', 'y', 'z'])

        self.gt_tracklets = dict((f, ObsCollection()) for f in self.list_frames)
        self.ignored_locs = dict((f, None) for f in self.list_frames)

        # Get the ground truth coordinates and identity
        for frame, objectID, xmin, ymin, width, height, _, _, _ in zip(df_gt.frames.values, df_gt.objectIDs.values, df_gt.xmin.values,
                                                                       df_gt.ymin.values, df_gt.width.values, df_gt.height.values, df_gt.x.values, df_gt.y.values, df_gt.z.values):

            locator = Box(coordinates=[xmin, ymin, xmin+width-1, ymin+height-1])
            observation = Observation(objectID=objectID, locator=locator, classe='pedestrian', score=1.0, frame=frame, flag='_GT_')

            self.gt_tracklets[frame].add_observation(observation)

        # Get the ignored area. Here, DanceTrack does not provide any ignored area.
        bin_map = np.zeros((self.height, self.width), dtype=bool)
        empty_locator = Mask.from_binmap(tl=Point(0, 0), bin_map=bin_map, img_width=self.width, img_height=self.height)

        for frame in self.list_frames:
            self.ignored_locs[frame] = empty_locator

    def get_path_image(self, frame: int):
        return Path(f"{PATH_DATA}/DanceTrack/{self.subset}/{self.scene_name}/img1/{frame:08d}.jpg")

    def save_benchmark(self, tracklets: Dict[int, ObsCollection], **kwargs):

        list_results = []

        for frame, obsCollection in tracklets.items():
            for obs in obsCollection:

                # Add a new row
                list_results += [{'frame':    frame,
                                  'objectID': obs.objectID,
                                  'xmin': obs.locator.xmin,
                                  'ymin': obs.locator.ymin,
                                  'width': obs.locator.width,
                                  'height': obs.locator.height,
                                  'score': f"{obs.score:0.3f}",
                                  'x:': -1, 'y': -1, 'z': -1
                                  }]

        df = pd.DataFrame(list_results)
        df.to_csv(f"{kwargs.get('folder')}/{self.scene_name}.txt", header=False, index=False, sep=',')

    def save_benchmark_from_Track(self, memory: Dict[int, Track], **kwargs):

        list_results = []

        for objectID, track in memory.items():
            for obs in track:

                # Add a new row
                list_results += [{'frame': obs.frame,
                                  'objectID': objectID,
                                  'xmin': obs.locator.xmin,
                                  'ymin': obs.locator.ymin,
                                  'width': obs.locator.width,
                                  'height': obs.locator.height,
                                  'score': f"{obs.score:0.3f}",
                                  'x:': 1, 'y': 1, 'z': 1
                                  }]

        df = pd.DataFrame(list_results)
        df.to_csv(f"{kwargs.get('folder')}/{self.scene_name}.txt", header=False, index=False, sep=',')

    def load_benchmarks(self, filename):
        
        # Open the file with bounding boxes
        dets = pd.read_csv(filename, names=['frames', 'objectIDs', 'xmin', 'ymin', 'width', 'height', 'scores', 'x', 'y', 'z'], sep=',')

        tracklets = dict((f, ObsCollection()) for f in self.list_frames)
        for frame, objectID, xmin, ymin, width, height, score in zip(dets.frames.values, dets.objectIDs.values, dets.xmin.values, dets.ymin.values,
                                                                     dets.width.values, dets.height.values, dets.scores.values):

            locator = Box(coordinates=[xmin, ymin, xmin+width-1, ymin+height-1])
            observation = Observation(objectID=objectID, locator=locator, classe='pedestrian', score=score, frame=frame, flag='_X_')
            tracklets[frame].add_observation(observation)

        return tracklets
    
    def evaluate_performance(self, **kwargs):
        bashCommand = f"python {PATH_BASE}/src/evaluation/DanceTrack/TrackEval/scripts/run_mot_challenge.py \
            --GT_FOLDER {PATH_DATA}/DanceTrack/{kwargs.get('split_to_eval')} \
            --SEQMAP_FILE {PATH_BASE}/src/evaluation/DanceTrack/dancetrack/{kwargs.get('split_to_eval')}_seqmap.txt \
            --SKIP_SPLIT_FOL True \
            --METRICS HOTA CLEAR Identity \
            --SPLIT_TO_EVAL {kwargs.get('split_to_eval')} \
            --TRACKERS_FOLDER {kwargs.get('trackers_folder')}/{self.dataset_name}-{self.subset} \
            --TRACKERS_TO_EVAL {kwargs.get('trackers_to_eval')} \
            --USE_PARALLEL=True \
            --TRACKER_SUB_FOLDER {kwargs.get('sub_folder')} \
            --OUTPUT_SUB_FOLDER {kwargs.get('sub_folder')}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        _, _ = process.communicate()

        # Print the results
        for classe in self.dict_COI.values():
            filename = Path(f"{kwargs['trackers_folder']}/{self.dataset_name}-{self.subset}/{kwargs['trackers_to_eval']}/{kwargs['sub_folder']}/{classe}_summary.txt")
            df = pd.read_csv(filename, sep=' ')
            logger.success(f"[{self.dataset_name}-{self.subset}-{classe}]  HOTA: {df['HOTA'][0]} DetA: {df['DetA'][0]} AssA: {df['AssA'][0]} MOTA: {df['MOTA'][0]} IDF1: {df['IDF1'][0]}")

   