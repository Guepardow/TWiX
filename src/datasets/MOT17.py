import subprocess
import pandas as pd
from typing import Dict
from pathlib import Path
from loguru import logger

from structures.locator import Box
from structures.track import Track
from structures.observation import Observation
from structures.baseDataset import BaseDataset
from structures.obsCollection import ObsCollection

PATH_CURRENT = Path(__file__).resolve().parent
PATH_DATA = PATH_CURRENT.parent.parent / 'data'
PATH_BASE = PATH_CURRENT.parent.parent


class MOT17(BaseDataset):
    def __init__(self, subset: str):

        self.list_subsets = ['test', 'train', 'train_half', 'val_half']
        assert subset in self.list_subsets, f"subset {subset} is not in {self.list_subsets}"

        super().__init__('MOT17', subset)
        
        self.dict_COI = {1: 'pedestrian'}
        self.framestep = 1
        self.level = 'box'
        
        if subset in ['train', 'train_half', 'val_half']:
            self.list_scenes = sorted(list(set([folder.name[:8] for folder in Path(f"{PATH_DATA}/MOT17/train").iterdir()])))        
        elif subset in ['test']:
            self.list_scenes = sorted(list(set([folder.name[:8] for folder in Path(f"{PATH_DATA}/MOT17/test").iterdir()])))   

    def load_scene(self, scene_name):

        if scene_name in ['MOT17-01', 'MOT17-02', 'MOT17-03', 'MOT17-04', 'MOT17-07', 'MOT17-08', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-12']:
            self.height, self.width = 1080, 1920
            self.fps = 30
        elif scene_name in ['MOT17-05', 'MOT17-06']:
            self.height, self.width = 480, 640
            self.fps = 14
        elif scene_name in ['MOT17-13', 'MOT17-14']:
            self.height, self.width = 1080, 1920
            self.fps = 25
        else:
            raise NotImplementedError(f"Scene {scene_name} is not recognized! Acceptable scene names are: \n{self.list_scenes}")

        self.scene_name = scene_name

        # List of images
        # For half version, get the code from CenterTrack: https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/convert_mot_to_coco.py
        if self.subset in ['train', 'test']:
            self.list_frames = sorted([int(f.name[:-4]) for f in Path(f"{PATH_DATA}/MOT17/{self.subset}/{self.scene_name}-DPM/img1").iterdir()])

        elif self.subset == 'train_half':
            all_train_frames = sorted([int(f.name[:-4]) for f in Path(f"{PATH_DATA}/MOT17/train/{self.scene_name}-DPM/img1").iterdir()])
            self.list_frames = all_train_frames[:len(all_train_frames)//2]

        elif self.subset == 'val_half':
            all_train_frames = sorted([int(f.name[:-4]) for f in Path(f"{PATH_DATA}/MOT17/train/{self.scene_name}-DPM/img1").iterdir()])
            self.list_frames = all_train_frames[len(all_train_frames)//2:]

    def load_benchmarks(self, filename):

        data = pd.read_csv(filename, names=['frames', 'objectIDs', 'xmin', 'ymin', 'width', 'height', 'scores', 'x', 'y', 'z'], sep=',')
        tracklets = dict((f, ObsCollection()) for f in self.list_frames)

        for frame, objectID, xmin, ymin, width, height, score in zip(data.frames.values, data.objectIDs.values, data.xmin.values, data.ymin.values, 
                                                                           data.width.values, data.height.values, data.scores.values):

            locator = Box(coordinates=[xmin, ymin, xmin+width-1, ymin+height-1])
            observation = Observation(objectID=objectID, locator=locator, classe='pedestrian', score=score, frame=frame, flag='_X_')
            tracklets[frame].add_observation(observation)

        return tracklets

    def load_oracle_infos(self):

        # Open the file with bounding boxes

        # MOT17 does not contain any official validation set: there is only train set
        df_gt = pd.read_csv(Path(f"{PATH_DATA}/MOT17/train/{self.scene_name}-DPM/gt/gt.txt"),
                            names=['frame', 'objectIDs', 'xmin', 'ymin', 'width', 'height', 'x', 'y', 'z'])
        df_gt = df_gt[df_gt['frame'].isin(self.list_frames)]  # filter frames out of the scene in case of subset == '..._half'

        self.gt_tracklets = dict((f, ObsCollection()) for f in self.list_frames)
        dict_ignored_obsv = dict((f, ObsCollection()) for f in self.list_frames)
        self.ignored_locs = dict((f, None) for f in self.list_frames)

        for frame, objectID, xmin, ymin, width, height, x, y in zip(df_gt.frame.values, df_gt.objectIDs.values, df_gt.xmin.values, df_gt.ymin.values,
                                                                       df_gt.width.values, df_gt.height.values, df_gt.x.values, df_gt.y.values):

            locator = Box(coordinates=[xmin, ymin, xmin+width-1, ymin+height-1])

            flag = '_GT_' if (x != 0) & (y == 1) else '_IGN_'
            # According to the devkit, when x=0, it is a 0-marked GT ; when y != 1, it is not a pedestrian
            # Code following https://bitbucket.org/amilan/motchallenge-devkit/src/default/evaluateTracking.m

            if flag == '_GT_':
                observation = Observation(objectID=objectID, locator=locator, classe='pedestrian', score=1.0, frame=frame, flag=flag)
                self.gt_tracklets[frame].add_observation(observation)
            elif flag == '_IGN_':
                observation = Observation(objectID=0, locator=locator, classe=None, score=None, frame=frame, flag=flag)
                dict_ignored_obsv[frame].add_observation(observation)

        # Get the ignored area per frame
        for frame in self.list_frames:
            self.ignored_locs[frame] = dict_ignored_obsv[frame].get_mask_ignored(img_width=self.width, img_height=self.height)

    def get_path_image(self, frame):

        if self.subset in ['train', 'train_half', 'val_half']:
            return Path(f"{PATH_DATA}/MOT17/train/{self.scene_name}-DPM/img1/{frame:06d}.jpg")
        else:
            return Path(f"{PATH_DATA}/MOT17/test/{self.scene_name}-DPM/img1/{frame:06d}.jpg")

    def save_benchmark(self, tracklets: Dict[int, ObsCollection], **kwargs):
        """
        Instructions from https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-format.txt

        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        The conf value contains the detection confidence in the det.txt files. For the ground truth, it acts as a flag whether the entry is to be considered.
        A value of 0 means that this particular instance is ignored in the evaluation, while any other value can be used to mark it as active.
        For submitted results, all lines in the .txt file are considered. The world coordinates x,y,z are ignored for the 2D challenge and can be filled with -1.
        Similarly, the bounding boxes are ignored for the 3D challenge. However, each line is still required to contain 10 values.

        All frame numbers, target IDs and bounding boxes are 1-based. Here is an example:

        Tracking with bounding boxes
        (MOT15, MOT16, MOT17, MOT20)
          1, 3, 794.27, 247.59, 71.245, 174.88, -1, -1, -1, -1
          1, 6, 1648.1, 119.61, 66.504, 163.24, -1, -1, -1, -1
          1, 8, 875.49, 399.98, 95.303, 233.93, -1, -1, -1, -1
          ...
        """

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
        if self.subset in ['train', 'test']:
            for detection_name in ['DPM', 'FRCNN', 'SDP']:
                df.to_csv(f"{kwargs.get('folder')}/{self.scene_name}-{detection_name}.txt", header=False, index=False, sep=',')
        elif self.subset in ['train_half', 'val_half']:
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
                                  'x:': -1, 'y': -1, 'z': -1
                                  }]

        df = pd.DataFrame(list_results)

        if self.subset in ['train', 'test']:
            for detection_name in ['DPM', 'FRCNN', 'SDP']:
                df.to_csv(f"{kwargs.get('folder')}/{self.scene_name}-{detection_name}.txt", header=False, index=False, sep=',')
        elif self.subset in ['train_half', 'val_half']:
            df.to_csv(f"{kwargs.get('folder')}/{self.scene_name}.txt", header=False, index=False, sep=',')

    def evaluate_performance(self, **kwargs):
        bashCommand = f"python {PATH_BASE}/src/evaluation/TrackEval/scripts/run_mot_challenge.py \
            --GT_FOLDER {PATH_BASE}/src/evaluation/TrackEval/data/gt/mot_challenge \
            --SEQMAP_FOLDER {PATH_BASE}/src/evaluation/TrackEval/data/gt/mot_challenge/seqmaps \
            --METRICS HOTA CLEAR Identity \
            --SPLIT_TO_EVAL {kwargs.get('split_to_eval')} \
            --TRACKERS_FOLDER {kwargs.get('trackers_folder')} \
            --TRACKERS_TO_EVAL {kwargs.get('trackers_to_eval')} \
            --GT_LOC_FORMAT {{gt_folder}}/{{seq}}/gt/gt.txt \
            --USE_PARALLEL=True \
            --OUTPUT_FOLDER {kwargs.get('trackers_folder')}/MOT17-{kwargs.get('split_to_eval')} \
            --TRACKER_SUB_FOLDER {kwargs.get('sub_folder')} \
            --OUTPUT_SUB_FOLDER {kwargs.get('sub_folder')}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        _, _ = process.communicate()

        # Print the results
        for classe in self.dict_COI.values():
            filename = Path(kwargs['trackers_folder']) / f"MOT17-{kwargs['split_to_eval']}" / kwargs['trackers_to_eval'] / kwargs['sub_folder'] / f'{classe}_summary.txt'
            df = pd.read_csv(filename, sep=' ')
            logger.success(f"[{self.dataset_name}-{self.subset}-{classe}]  HOTA: {df['HOTA'][0]} DetA: {df['DetA'][0]} AssA: {df['AssA'][0]} MOTA: {df['MOTA'][0]} IDF1: {df['IDF1'][0]}")

   