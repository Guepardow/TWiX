import subprocess
import pandas as pd
from typing import Dict
from pathlib import Path
from loguru import logger

from structures.track import Track
from structures.locator import Box
from structures.observation import Observation
from structures.obsCollection import ObsCollection
from structures.baseDataset import BaseDataset

PATH_CURRENT = Path(__file__).resolve().parent
PATH_DATA = PATH_CURRENT.parent.parent / 'data'
PATH_BASE = PATH_CURRENT.parent.parent


class KITTIMOT(BaseDataset):
    def __init__(self, subset: str):

        self.list_subsets = ['training_minus_val', 'val', 'test', 'training']
        assert subset in self.list_subsets, f"subset {subset} is not in {self.list_subsets}"

        super().__init__('KITTIMOT', subset)

        self.dict_COI = {1: 'pedestrian', 3: 'car'}        
        self.framestep = 1
        self.level = 'box'

        list_scenes_train = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
        list_scenes_val = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
        if self.subset == 'training_minus_val':
            self.list_scenes = list_scenes_train
        elif self.subset == 'val':
            self.list_scenes = list_scenes_val
        elif self.subset == 'training':
            self.list_scenes = sorted(list_scenes_train + list_scenes_val)
        elif self.subset == 'test':
            self.list_scenes = sorted([path.name for path in Path(f"{PATH_DATA}/KITTIMOT/data_tracking_image_2/testing/image_02").iterdir()])

    def load_scene(self, scene_name):

        if scene_name not in self.list_scenes:
            raise KeyError(f"The scene {scene_name} is not a valid scene name! Acceptable scene names are: \n{self.list_scenes}")
        self.scene_name = scene_name

        self.fps = 10

        if self.subset in ['training_minus_val', 'val', 'training']:
            if scene_name in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013']:
                self.width, self.height = 1242, 375
            elif scene_name in ['0014', '0015', '0016', '0017']:
                self.width, self.height = 1224, 370
            elif scene_name in ['0018', '0019']:
                self.width, self.height = 1238, 374
            elif scene_name in ['0020']:
                self.width, self.height = 1241, 376
            else:
                raise NotImplementedError(f"The scene {scene_name} is not recognised in KITTIMOT-{self.subset}!")
        elif self.subset == 'test':
            if scene_name in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016']:
                self.width, self.height = 1242, 375
            elif scene_name in ['0017', '0018', '0019', '0020', '0021', '0022', '0023', '0024', '0025', '0026']:
                self.width, self.height = 1224, 370
            elif scene_name in ['0027', '0028']:
                self.width, self.height = 1226, 370
            else:
                raise NotImplementedError(f"The scene {scene_name} is not recognised in KITTIMOT-{self.subset}!")

        if self.subset in ['training_minus_val', 'val', 'training']:
            
            self.list_frames = sorted([int(frame.name[:-4]) for frame in Path(f"{PATH_DATA}/KITTIMOT/data_tracking_image_2/training/image_02/{scene_name}").iterdir()])
        elif self.subset == 'test':
            self.list_frames = sorted([int(frame.name[:-4]) for frame in Path(f"{PATH_DATA}/KITTIMOT/data_tracking_image_2/testing/image_02/{scene_name}").iterdir()])

    def load_benchmarks(self, filename):

        dict_classe = {'Car': 'car', 'Pedestrian': 'pedestrian'}

        df = pd.read_csv(filename, sep=' ', header=None)
        df.columns = ['frame', 'objectID', 'classe', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'height3D', 'width3D', 'length3D', 'x3D', 'y3D', 'z3D', 'rotation_y', 'score']
        zip_names = zip(df.frame.values, df.objectID.values, df.classe.values, df.truncated.values, df.occluded.values, df.alpha.values, df.xmin.values, df.ymin.values, df.xmax.values, df.ymax.values, df.height3D.values, df.width3D.values, df.length3D.values, df.x3D.values, df.y3D.values, df.z3D.values, df.rotation_y.values, df.score.values)

        tracklets = dict((f, ObsCollection()) for f in self.list_frames)

        for frame, objectID, classe, _, _, _, xmin, ymin, xmax, ymax, _, _, _, _, _, _, _, score in zip_names:

            locator = Box([xmin, ymin, xmax, ymax])
            observation = Observation(objectID=objectID, locator=locator, classe=dict_classe[classe], score=score, frame=frame, flag='_X_')
            tracklets[frame].add_observation(observation)

        return tracklets

    def load_oracle_infos(self):
        # More information : http://www.cvlibs.net/datasets/kitti/eval_tracking.php
        # More information : https://github.com/JonathonLuiten/TrackEval/blob/master/docs/KITTI-format.txt

        dict_classe = {'Car': 'car', 'Van': None, 'Truck': None,
                       'Pedestrian': 'pedestrian', 'Person_sitting': None,
                       'Cyclist': None, 'Tram': None, 'Misc': None, 'DontCare': None}  # same names as COCO

        self.gt_tracklets = dict((f, ObsCollection()) for f in self.list_frames)
        dict_ignored_obsv = dict((f, ObsCollection()) for f in self.list_frames)
        self.ignored_locs = dict((f, None) for f in self.list_frames)

        # Read the file containing the ground truth masks
        assert self.subset in ('training_minus_val', 'val', 'training'), "The ground truth annotations are not available!"

        df_gt = pd.read_table(Path(f"{PATH_DATA}/KITTIMOT/data_tracking_label_2/training/label_02/{int(self.scene_name):04d}.txt"), sep=' ',
                              header=None, names=['frame', 'id', 'class_id', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', '3Dheight', '3Dwidth', '3Dlength', '3Dx', '3Dy', '3Dz', 'rotation_y'])
        df_gt = df_gt.sort_values(by='frame')  # Now, consecutive rows may belong to the same frame

        for gt_frame, gt_id, gt_classe, xmin, ymin, xmax, ymax in zip(df_gt.frame.values, df_gt.id.values, df_gt.class_id.values, df_gt.xmin.values, df_gt.ymin.values, df_gt.xmax.values, df_gt.ymax.values):

            gt_classe = dict_classe.get(gt_classe)
            locator = Box([xmin, ymin, xmax, ymax])  # ymax instead of ymax-1

            if gt_classe is None:  # It is a ignored region
                new_observation = Observation(objectID=0, locator=locator, classe=gt_classe, score=1.0, frame=gt_frame, flag='_IGN_')
                dict_ignored_obsv[gt_frame].add_observation(new_observation)

            else:
                new_observation = Observation(objectID=gt_id, locator=locator, classe=gt_classe, score=1.0, frame=gt_frame, flag='_GT_')
                self.gt_tracklets[gt_frame].add_observation(new_observation)

        # Get the ignored area per frame
        for frame in self.list_frames:
            self.ignored_locs[frame] = dict_ignored_obsv[frame].get_mask_ignored(img_width=self.width, img_height=self.height)
     
    def get_path_image(self, frame: int):
        if self.subset in ['training_minus_val', 'val', 'training']:
            return Path(f"{PATH_DATA}/KITTIMOT/data_tracking_image_2/training/image_02/{self.scene_name}/{frame:06d}.png")
        elif self.subset == 'test':
            return Path(f"{PATH_DATA}/KITTIMOT/data_tracking_image_2/testing/image_02/{self.scene_name}/{frame:06d}.png")

    def save_benchmark(self, tracklets: Dict[int, ObsCollection], **kwargs):
        """
        Get the metrics for KITTIMOT : https://github.com/JonathonLuiten/TrackEval/blob/master/docs/KITTI-format.txt
        """
        dict_classe = {'car': 'Car', 'pedestrian': 'Pedestrian'}
        list_results = []

        for f, obsCollection in tracklets.items():

            for obs in obsCollection:

                # Add a new row
                list_results += [{'frame':    f,
                                  'objectID': int(obs.objectID),
                                  'classe': dict_classe.get(obs.classe),
                                  'truncated': 0, 'occluded': 0, 'alpha': 0,
                                  'xmin': obs.locator.xmin, 'ymin': obs.locator.ymin,
                                  'xmax': obs.locator.xmax, 'ymax': obs.locator.ymax,
                                  '3Dheight': 0, '3Dwidth': 0, '3Dlength': 0,
                                  '3Dx': 0, '3Dy': 0, '3Dz': 0,
                                  'rotation_y': 0,
                                  'score': f"{obs.score:0.3f}"}]

        df = pd.DataFrame(list_results)
        df.to_csv(f"{kwargs.get('folder')}/{self.scene_name}.txt", header=False, index=False, sep=' ')

    def save_benchmark_from_Track(self, memory: Dict[int, Track], **kwargs):
        """
        No possibility to put a Tracker type in the input since it is based on Scene ...
        """

        dict_classe = {'car': 'Car', 'pedestrian': 'Pedestrian'}
        map_results = [{'frame':    obs.frame,
                        'objectID': int(objectID),
                        'classe':   dict_classe.get(track.classe),
                        'truncated': 0, 'occluded': 0, 'alpha': 0,
                        'xmin': obs.locator.xmin, 'ymin': obs.locator.ymin,
                        'xmax': obs.locator.xmax, 'ymax': obs.locator.ymax,
                        '3Dheight': 0, '3Dwidth': 0, '3Dlength': 0,
                        '3Dx': 0, '3Dy': 0, '3Dz': 0,
                        'rotation_y': 0,
                        'score': f"{obs.score:0.3f}"}
                       for objectID, track in memory.items() for obs in track]

        df = pd.DataFrame(map_results)
        df.to_csv(f"{kwargs.get('folder')}/{self.scene_name}.txt", header=False, index=False, sep=' ')

    def evaluate_performance(self, **kwargs):
        bashCommand = f"python {PATH_BASE}/src/evaluation/TrackEval/scripts/run_kitti.py \
            --GT_FOLDER {PATH_BASE}/src/evaluation/TrackEval/data/gt/kitti/kitti_2d_box_train \
            --METRICS HOTA CLEAR Identity \
            --SPLIT_TO_EVAL {kwargs.get('split_to_eval')} \
            --TRACKERS_FOLDER {kwargs.get('trackers_folder')}/{self.dataset_name}-{self.subset} \
            --TRACKERS_TO_EVAL {kwargs.get('trackers_to_eval')} \
            --USE_PARALLEL=True \
            --TRACKER_SUB_FOLDER {kwargs.get('sub_folder')} \
            --OUTPUT_SUB_FOLDER {kwargs.get('sub_folder')}"            
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        _, _ = process.communicate()

        # Print the results for all classes
        for classe in self.dict_COI.values():
            filename = Path(f"{kwargs['trackers_folder']}/{self.dataset_name}-{self.subset}/{kwargs['trackers_to_eval']}/{kwargs['sub_folder']}/{classe}_summary.txt")
            df = pd.read_csv(filename, sep=' ')
            logger.success(f"[{self.dataset_name}-{self.subset}-{classe}]  HOTA: {df['HOTA'][0]} DetA: {df['DetA'][0]} AssA: {df['AssA'][0]} MOTA: {df['MOTA'][0]} IDF1: {df['IDF1'][0]}")