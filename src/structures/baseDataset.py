import cv2
import numpy as np
import pandas as pd
import prettytable
from loguru import logger
from typing import Union, Dict
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt

from .observation import Observation
from .obsCollection import ObsCollection
from .locator import Point, Box, Mask


class BaseDataset:
    """
    A class used to describe a video sequence
    """

    def __init__(self, dataset_name: str, subset: str):
        """
        :param dataset_name: name of the dataset
        :param subset: part of the subset
        """
        self.dataset_name = dataset_name  # Name of the dataset (e.g. KITTIMOTS)
        self.subset = subset              # Part of the data (e.g. train)

        # Properties initialized with the dataset
        self.list_subsets = None          # All available subsets (e.g. ['train', 'val', 'test'])
        self.list_scenes = None           # All video sequences names (e.g. ['0001', '0002', '0003'])
        self.level = None                 # Precision on the locator for evaluation (e.g. 'box')
        self.dict_COI = None              # Dictionnary of objects of interest (e.g. {1: 'pedestrian', 3: 'car'})
        self.framestep = None             # Incremental value in the frames' names (e.g. 1)

        # Properties initialized with load_scene()
        self.scene_name = None            # Name of the loaded scene (e.g. '0002')
        self.width = None                 # Width of images of the loaded scene (e.g. 1920)
        self.height = None                # Height of images of the loaded scene (e.g. 1080)
        self.fps = None                   # Frame per second of the loaded scene (e.g. 25)
        self.list_frames = None           # All frames of the scene (e.g. [1, 2, 3, 4, 5, 6, 7, 8])

        # Property initialized with load_oracle_infos()
        self.gt_tracklets = None          # Dictionnary of ObsCollection with the ground truth observation per frame
        self.ignored_locs = None          # Dictionnary of Locator with the ignored regions per frame

    @property
    def first_frame(self):
        return min(self.list_frames) if self.list_frames is not None else None
    
    @property
    def last_frame(self):
        return max(self.list_frames) if self.list_frames is not None else None
    
    @property
    def dict_rev_COI(self):
        # Reversed dictionnary of objects of interest (e.g. {'pedestrian': 1, 'car': 3})
        return {v: k for k, v in self.dict_COI.items()} if self.dict_COI is not None else None

    def load_scene(self, scene_name: str):
        # This is specific to a dataset
        raise NotImplementedError(f"The function load_scene is not implemented yet for {self.dataset_name}!")

    def get_path_image(self, frame: int) -> str:
        # This is specific to a dataset
        raise NotImplementedError(f"The function get_path_image is not implemented yet for {self.dataset_name}!")

    def load_oracle_infos(self):
        # This is specific to a dataset
        raise NotImplementedError(f"The function load_oracle_infos is not implemented yet for {self.dataset_name}!")

    def load_benchmarks(self, filename: str):
        # This is specific to a dataset
        raise NotImplementedError(f"The function load_benchmarks is not implemented yet for {self.dataset_name}!")

    def save_benchmark(self, tracklets: Dict[int, ObsCollection], path_to_file: str):
        # This is specific to a dataset
        raise NotImplementedError(f"The function save_benchmark is not implemented yet for {self.dataset_name}!")

    def evaluate_performance(self, **kwargs):
        # This is specific to a dataset
        raise NotImplementedError(f"The function evaluate_performance is not implemented yet for {self.dataset_name}!")

    def unload_oracle_infos(self):
        self.gt_tracklets = None
        self.ignored_locs = None

    def unload_scene(self):
        self.scene_name = None
        self.width = None
        self.height = None
        self.fps = None
        self.list_frames = None

        self.unload_oracle_infos()

    def viz(self, frame: int):
        """
        Visualize the image at a specific frame

        :param frame: frame of interest
        """

        image = cv2.imread(self.get_path_image(frame))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def show(self, frame: int, size: float = 2):
        """
        Visualize the image at a specific frame

        :param frame: frame of interest
        :param size: size of the image when plotted
        """

        image = cv2.imread(self.get_path_image(frame))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(self.height/100*size, self.width/100*size))
        plt.imshow(image)

    def viz_oracle(self, frame: int, show_boxes: bool = True, show_contours: bool = True, show_masks: bool = True, n_colors: int = 10, size: float = 5) -> np.ndarray:
        """
        Visualize the ground truth for a specific frame, with ignored regions

        :param frame: frame of interest
        :param show_boxes: if True, show the bounding boxes
        :param show_contours: if True, show the contours of segmentation masks
        :param show_masks: if True, show the segmentation masks
        :param n_colors: number of colors of objects
        :param size: size of the image when plotted
        """
        are_tracklets_absent = self.gt_tracklets is None

        if are_tracklets_absent:
            self.load_oracle_infos()
            logger.warning('Loads the oracle infos: load them once if you want to visualize faster for multiple frames.')

        # Data from the ground truth
        gt_obsColl = self.gt_tracklets.get(frame)
        image = self.viz_obsCollection(frame, gt_obsColl, show_masks=show_masks, show_contours=show_contours, show_boxes=show_boxes, show_texts=True, n_colors=n_colors, only_locators=False)

        # Ignored regions as a binary mask
        mask_ignored_loc = self.ignored_locs[frame].to_binmap(img_width=self.width, img_height=self.height)

        # Create an RGBA array for the overlay
        overlay = np.zeros((mask_ignored_loc.shape[0], mask_ignored_loc.shape[1], 4), dtype=np.uint8)
        overlay[:, :, 3] = mask_ignored_loc * 255  # Alpha channel for transparency

        # Combine the image and overlay using masking
        alpha = 0.75  # transparency
        masked_image = image.copy()
        masked_image[mask_ignored_loc == 1] = (1 - alpha) * image[mask_ignored_loc == 1] + alpha * overlay[mask_ignored_loc == 1, :3]
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

        if are_tracklets_absent:
            self.unload_oracle_infos()

        return masked_image
    
    def show_oracle(self, frame: int, show_boxes: bool = True, show_contours: bool = True, show_masks: bool = True, n_colors: int = 10, size: float = 5):
        
        image = self.viz_oracle(frame, show_boxes, show_contours, show_masks, n_colors, size)

        plt.figure(figsize=(self.height/100*size, self.width/100*size))
        plt.imshow(image)


    def show_locator(self, frame: int, locator: Union[Point, Box, Mask], size: float = 5, show_masks=True, show_contours=True, show_boxes=False, zoom=False, only_locator=False):

        if isinstance(locator, (Point, Box, Mask)):

            image = self.viz_locator(frame, locator, show_masks, show_contours, show_boxes, zoom, only_locator)

            plt.figure(figsize=(self.height/100*size, self.width/100*size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)

        else:
            raise NotImplementedError(f"The type of the locator is not recognized: {type(locator)}")

    def viz_locator(self, frame: int, locator: Union[Point, Box, Mask], show_mask: bool, show_contour: bool, show_box: bool, zoom: bool, only_locator: bool) -> np.ndarray:

        # Open the current image from the folder
        if only_locator:
            image = np.zeros((self.height, self.width, 3), dtype='uint8')
        else:
            image = cv2.imread(self.get_path_image(frame))

        # Draw a rectangle or a point
        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        if isinstance(locator, (Box, Mask)):
            xmin, ymin, xmax, ymax = locator.coordinates
        elif isinstance(locator, Point):
            xmin, ymin, xmax, ymax = locator.x, locator.y, locator.x, locator.y

        if show_box & (not zoom):
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)

        # Draw a transparent mask if the locator is a Mask
        if isinstance(locator, Mask):

            # Part of the image with the object of interest
            sub_frame = image[locator.tl.y:(locator.tl.y+locator.height), locator.tl.x:(locator.tl.x+locator.width)]

            # Binary mask on RGB channels
            binary_mask = np.repeat(locator.bin_map[:, :, np.newaxis], 3, axis=2)
            sub_mask = binary_mask.astype('uint8') * np.array([255, 255, 255], dtype='uint8')

            # Addition to both part on the region of interest
            overlay = cv2.addWeighted(sub_frame, 1.0, sub_mask, 0.5, 0.0)

            # Fusion
            if show_mask:
                image[locator.tl.y:(locator.tl.y+locator.height), locator.tl.x:(locator.tl.x+locator.width)] = overlay

            if show_contour:
                contours, _ = cv2.findContours(locator.to_binmap(self.width, self.height).astype('uint8'), 1, 2)
                image = cv2.drawContours(image, contours, -1, (255, 255, 255), 2)

        cv2.putText(image, str(frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)  # counter of frame

        if zoom:
            return image[ymin:(ymax+1), xmin:(xmax+1), :]
        else:
            return image

    def show_object(self, obsColl: ObsCollection):
        assert len(set([obs.objectID for obs in obsColl])) <= 1, "The collection contains observations from multiple objects!"

        def get_dimensions(n, max_cols):
            """
            Optimal number of rows and columns
            """
            if n <= max_cols:  # 1 row, n columns
                return 1, n
            else:
                divisors = [x for x in range(n, 1, -1) if n % x == 0]
                for d in divisors:
                    if (d <= max_cols) and (n/d**2 > 4):
                        return n // d, d
                    else:
                        return int(np.ceil(n/5)), 5

        def get_max_view(x1, y1, x2, y2, width_max, height_max, scene_width, scene_height):
            """
            Return the zooming view of a Box with a small padding on the border

            x1: minimal value of x
            x2: maximal value of x
            y1: minimal value of y
            y2: maximal value of y
            width_max: maximal width
            height_max: maximal height
            """
            width = x2 - x1 + 1
            height = y2 - y1 + 1

            if (width == width_max) and (height == height_max):
                return x1, y1, x2, y2

            # We have to pad
            pad_x = (width_max - width) // 2
            pad_y = (height_max - height) // 2

            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y

            # Be careful to be inside of the image
            if x1 < 0:
                delta_x = -x1  # positive value
                x1, x2 = x1 + delta_x, x2 + delta_x

            if y1 < 0:
                delta_y = -y1  # positive value
                y1, y2 = y1 + delta_y, y2 + delta_y

            if x2 >= scene_width:
                delta_x = x2 - scene_width+1  # positive value
                x1, x2 = x1 - delta_x, x2 - delta_x

            if y2 >= scene_height:
                delta_y = y2 - scene_height+1  # positive value
                y1, y2 = y1 - delta_y, y2 - delta_y

            return x1, y1, x2, y2

        nrows, ncols = get_dimensions(len(obsColl), max_cols=5)

        # Same ratio for each image : resize the crop
        max_width = max([obs.locator.width for obs in obsColl])
        max_height = max([obs.locator.height for obs in obsColl])

        blank_image = np.ones((max_height, max_width, 3))

        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
        axes = axes.flatten()

        for idx, (obs, ax) in enumerate(zip(obsColl, axes)):

            image = cv2.imread(self.get_path_image(obs.frame))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xmin, ymin, xmax, ymax = obs.locator.coordinates
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)

            xmin, ymin, xmax, ymax = get_max_view(xmin, ymin, xmax, ymax, max_width, max_height, self.width, self.height)

            ax.imshow(image[ymin: (ymax+1), xmin:(xmax+1)])

            ax.set_xticks(np.linspace(0, max_width, 4, dtype=int))
            ax.set_yticks(np.linspace(0, max_height, 4, dtype=int))

            ax.set_xticklabels(np.linspace(xmin, xmax, 4, dtype=int))
            ax.set_yticklabels(np.linspace(ymin, ymax, 4, dtype=int))

            ax.set_title(f"Frame {obs.frame} ({100*obs.score:0.1f}%)")

        for idx in range(len(obsColl), nrows*ncols):
            axes[idx].imshow(blank_image)
            axes[idx].axis('off')

        plt.show()

    def show_obsCollection(self, frame: int, obsCollection: ObsCollection, show_masks=True, show_contours=True, show_boxes=True, show_texts=True, n_colors: int = 10, size: int = 5, only_locators=False):

        image = self.viz_obsCollection(frame, obsCollection, show_masks, show_contours, show_boxes, show_texts, n_colors, only_locators)

        plt.figure(figsize=(self.height/100*size, self.width/100*size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)

    def viz_obsCollection(self, frame: int, obsCollection: ObsCollection, show_masks: bool=True, show_contours: bool=True, show_boxes: bool=True, show_texts: bool=True, n_colors: int=10, only_locators: bool=False) -> np.ndarray:
        # NB : frame is needed if there is no object in the obsCollection!

        # Check that the collection contains only observations from the same frame
        assert len(set([obs.frame for obs in obsCollection])) <= 1, "The collection contains observations from multiple frames!"

        # Palette
        palette = sns.color_palette("bright", n_colors)  # number of different colors

        # Open the current image from the folder
        if only_locators:
            image = np.zeros((self.height, self.width, 3), dtype='uint8')
        else:
            image = cv2.imread(self.get_path_image(frame))

        has_assigned_objectID = any(np.array([obs.objectID for obs in obsCollection]) != 0)
        for obs in obsCollection:

            if has_assigned_objectID:  # 0 indicates that it is a detection without any objectID. For DanceTrack, they labeled the ground with 0-index ...
                color = tuple(int(255 * i) for i in palette[obs.objectID % n_colors])
            else:
                color = (255, 255, 255)  # white color for just detected object without an assigned identity

            # Draw a rectangle or a point
            xmin, ymin, xmax, ymax = 0, 0, 0, 0
            if isinstance(obs.locator, (Box, Mask)):
                xmin, ymin, xmax, ymax = obs.locator.coordinates
            elif isinstance(obs.locator, Point):
                xmin, ymin, xmax, ymax = obs.locator.x, obs.locator.y, obs.locator.x, obs.locator.y

            if show_boxes:
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

            # Draw a transparent mask if the locator is a Mask
            if isinstance(obs.locator, Mask):

                # Color in an array
                if has_assigned_objectID:
                    color_array = np.array([(255 * i) for i in palette[obs.objectID % n_colors]]).astype('uint8')
                else:
                    color_array = np.array([255, 255, 255]).astype('uint8')

                # Part of the image with the object of interest
                sub_frame = image[obs.locator.tl.y:(obs.locator.tl.y+obs.locator.height), obs.locator.tl.x:(obs.locator.tl.x+obs.locator.width)]

                # Binary mask on RGB channels
                binary_mask = np.repeat(obs.locator.bin_map[:, :, np.newaxis], 3, axis=2)
                sub_mask = binary_mask.astype('uint8')*color_array

                # Addition to both part on the region of interest
                overlay = cv2.addWeighted(sub_frame, 1.0, sub_mask, 0.5, 0.0)

                # Fusion
                if show_masks:
                    image[obs.locator.tl.y:(obs.locator.tl.y+obs.locator.height), obs.locator.tl.x:(obs.locator.tl.x+obs.locator.width)] = overlay

                if show_contours:
                    contours, _ = cv2.findContours(obs.locator.to_binmap(self.width, self.height).astype('uint8'), 1, 2)
                    image = cv2.drawContours(image, contours, -1, color, 2)

            # Write locator's information
            x_pos = xmin + 5
            y_pos = ymin + 15 if ymin < 10 else ymin - 5

            if show_texts:
                if has_assigned_objectID:
                    text_locator = f"{obs.classe} {obs.objectID:.0f} ({100*obs.score:.1f}%)"
                else:
                    text_locator = f"{obs.classe} ({100*obs.score:.1f}%)"

                cv2.putText(image, text_locator, (int(x_pos), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.putText(image, str(frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)  # counter of frame

        return image

    @staticmethod
    def save_detections(filename: str, tracklets: Dict[int, ObsCollection], dict_classe: Dict[str, int]):
        """
        Save the detections of bounding boxes
        Format : <Timestep>(int), <Track ID>(int), <Class Number>(int), <Detection Confidence>(float), <Xmin>(float), <Ymin>(float), <Xmax>(float), <Ymax>(float)
        """
        list_results = []

        for frame, obsColl in tracklets.items():
            for obs in obsColl:

                # Add a new row
                list_results += [{'frame':    frame,
                                  'objectID': obs.objectID,
                                  'classe':   dict_classe[obs.classe],
                                  'score':    f"{obs.score:0.3f}",
                                  'xmin':     obs.locator.xmin,
                                  'ymin':     obs.locator.ymin,
                                  'xmax':     obs.locator.xmax,
                                  'ymax':     obs.locator.ymax}]

        df = pd.DataFrame(list_results)
        df.to_csv(f'{filename}', header=False, index=False, sep=' ')

    def load_detections(self, filename: str, dict_classe: Dict[int, str]) -> Dict[int, ObsCollection]:

        # Open the file with bounding boxes
        dets = pd.read_csv(filename, names=['frames', 'objectIDs', 'classes', 'scores', 'xmin', 'ymin', 'xmax', 'ymax'], sep=' ')

        tracklets = dict((f, ObsCollection()) for f in self.list_frames)
        for frame, objectID, classe, score, xmin, ymin, xmax, ymax in zip(dets.frames.values, dets.objectIDs.values, dets.classes.values, dets.scores.values,
                                                                          dets.xmin.values, dets.ymin.values, dets.xmax.values, dets.ymax.values):

            locator = Box(coordinates=[xmin, ymin, xmax, ymax])
            observation = Observation(objectID=objectID, locator=locator, classe=dict_classe[classe], score=score, frame=frame, flag='_DET_')
            tracklets[frame].add_observation(observation)

        return tracklets

    def __len__(self):
        return int((self.last_frame - self.first_frame)/self.framestep + 1)

    def __str__(self):
        text = f"The dataset {self.dataset_name}-{self.subset} contains {len(self.list_scenes)} scenes\n"
        text += f"Classes of interest ({len(self.dict_COI)}) : {self.dict_COI}\n"
        if self.scene_name is not None:
            text += f"Selected scene: {self.scene_name}\n"
            text += f"{len(self)} images of size ({self.width} x {self.height}) at {self.fps} FPS\n"
            return text
        else:

            scene_table = prettytable.PrettyTable(['name', '# frames', 'width', 'height', 'fps', 'duration'], hrules=prettytable.HEADER)
            for scene_name in self.list_scenes:
                self.load_scene(scene_name)
                duration = len(self.list_frames) / self.fps
                scene_table.add_row([self.scene_name, f"{len(self.list_frames):4.0f}", self.width, self.height, self.fps, f"{duration:3.1f} s"])

            self.unload_scene()
            return f"{text}\n{scene_table.get_string()}"

    def print_gt_stat(self):

        assert self.scene_name is None, f"A scene ({self.scene_name}) is loaded!"
        stat_table = prettytable.PrettyTable(['#', 'scene', '# frames', 'width', 'height', 'fps', 'duration', '#IDs', '#Obs', 'area'], hrules=prettytable.HEADER)

        total_frames, total_duration, total_n_ids, total_n_obs, global_areas, global_classes = 0, 0, 0, 0, [], []
        for idx, scene_name in enumerate(self.list_scenes, 1):
            self.load_scene(scene_name)
            self.load_oracle_infos()

            n_ids = len(set([obs.objectID for obsColl in self.gt_tracklets.values() for obs in obsColl]))
            n_obs = sum([len(obsColl) for obsColl in self.gt_tracklets.values()])

            classes = [obs.classe for obsColl in self.gt_tracklets.values() for obs in obsColl]
            areas = [obs.locator.area for obsColl in self.gt_tracklets.values() for obs in obsColl]
            area_avg = np.mean(areas)
            area_min, area_max = np.min(areas), np.max(areas)

            duration = len(self.list_frames) / self.fps
            stat_table.add_row([idx, self.scene_name, f"{len(self.list_frames):4.0f}", self.width, self.height, self.fps, f"{duration:3.1f} s", n_ids, n_obs, f"{area_avg:0.0f} [{area_min:0.0f}-{area_max:0.0f}]"])

            # Statistics on all scenes
            total_frames += len(self.list_frames)
            total_duration += duration
            total_n_ids += n_ids
            total_n_obs += n_obs
            global_areas.extend(areas)
            global_classes.extend(classes)

        global_area_avg, global_area_min, global_area_max = np.mean(global_areas), np.min(global_areas), np.max(global_areas)

        n_scenes = len(self.list_scenes)
        stat_table.add_row(['-', 'TOTAL', f"{total_frames:4.0f}", '-', '-', '-', f"{total_duration:3.1f} s", total_n_ids, total_n_obs, f"{global_area_avg:0.0f} [{global_area_min:0.0f}-{global_area_max:0.0f}]"])
        stat_table.add_row(['-', 'AVG/SCENE', f"{total_frames/n_scenes:4.0f}", '-', '-', '-', f"{total_duration/n_scenes:3.1f} s", f"{total_n_ids/n_scenes:3.1f}", f"{total_n_obs/n_scenes:3.1f}", '-'])
        stat_table.add_row(['-', 'AVG/FRAME', '-', '-', '-', '-', '-', '-', f"{total_n_obs/total_frames:3.1f}", '-'])

        self.unload_scene()
        print(f"{stat_table.get_string()}")

        counter = Counter(global_classes)
        for className, n_count in counter.items():
            print(f"Total of {className} GT observations: {n_count}")
