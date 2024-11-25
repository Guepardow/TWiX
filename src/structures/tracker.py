import cv2
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict, Optional

from .locator import Mask
from .baseDataset import BaseDataset
from .obsCollection import ObsCollection


class Tracker:

    def __init__(self, scene: BaseDataset, tracklets: Dict[int, ObsCollection]):

        # Create empty ObsCollection, necessary if there are empty collections
        self.tracklets = dict((f, ObsCollection()) for f in scene.list_frames)
        for f, obsColl in tracklets.items():
            self.tracklets[f] = obsColl  # {frame: obsCollection}
        # A Tracker contains each frame, even those where there is no object

        self.scene = scene

        all_objectIDs = [obs.objectID for obsCollection in list(self.tracklets.values()) for obs in obsCollection]
        self.max_objectID = max(all_objectIDs) if all_objectIDs else 0

        self.all_tracks = self.get_all_tracks()

    def get_new_identities(self, obsCollection: ObsCollection) -> ObsCollection:
        """
        Gives an new objectID to all observations of obsCollection (objectID == 0), taking into account the previous objectIDs

        :param obsCollection: collection of observations containing potential unassigned observation
        """
        for obs in obsCollection:
            if obs.objectID <= 0:  # not assigned identity
                obs.objectID = self.max_objectID + 1
                self.max_objectID += 1

        return obsCollection

    def get_all_tracks(self) -> Dict[int, ObsCollection]:

        all_tracks = {}
        for f, obsCollection in self.tracklets.items():
            for obs in obsCollection:
                if obs.objectID not in all_tracks.keys():
                    all_tracks[obs.objectID] = ObsCollection()
                all_tracks[obs.objectID].add_observation(obs)

        return all_tracks

    def show(self, frame: int, show_masks=True, show_contours=True, show_boxes=True, show_texts=True, n_colors: int = 10, size: int = 5, only_locators: bool = False):

        image = self.scene.viz_obsCollection(frame, self.tracklets[frame], show_masks, show_contours, show_boxes, show_texts, n_colors, only_locators)

        plt.figure(figsize=(self.scene.height/100*size, self.scene.width/100*size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)

    def visualize(self, path_save: str, show_masks=True, show_contours=True, show_boxes=True, show_texts=True, n_colors=10, only_locators=False):

        out = cv2.VideoWriter(path_save, cv2.VideoWriter_fourcc(*'XVID'), self.scene.fps, (self.scene.width, self.scene.height))

        for f, obsCollection in self.tracklets.items():

            image = self.scene.viz_obsCollection(f, obsCollection, show_masks, show_contours, show_boxes, show_texts, n_colors, only_locators)

            cv2.imshow('image', image)
            out.write(image)

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()

    def visualize_oracle(self, path_save: str, show_masks=True, show_contours=True, show_boxes=True, n_colors=10):

        out = cv2.VideoWriter(path_save, cv2.VideoWriter_fourcc(*'XVID'), self.scene.fps, (self.scene.width, self.scene.height))

        for f, obsCollection in self.tracklets.items():

            # viz_oracle contains the ignored regions depicted as black zones
            image = self.scene.viz_oracle(f, show_boxes, show_contours, show_masks, n_colors)

            cv2.imshow('image', image)
            out.write(image)

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()


    def get_gt_tracklets(self, gt_tracklets: Dict[int, ObsCollection], ignored_locs: Dict[int, Mask], threshold: float) -> Dict[int, Optional[int]]:
        """
        For each tracklet in the tracker, returns the ground truth objectID which corresponds the most with the collection of observations.
        This is based on a Track-IoU measure.
        TP : a positive integer
        FP : None
        ignored : -1

        :param gt_tracklets: tracklets containing the ground truth annotations
        :param ignored_tracklets: tracklets with the ignored regions
        :param threshold: minimal IoU to consider a valid match between two observations
        """

        dict_gt_objectID_tracklets = {}
        for frame, obsColl in self.tracklets.items():  # at the frame level

            dict_gt_objectID_frame = obsColl.get_gt_frame(gt_tracklets[frame], ignored_locs[frame], threshold=threshold,
                                                          img_width=self.scene.width, img_height=self.scene.height)
            # {0: 45, 1: 51, 2: None, 3: 46, 4: -1} (position: gt_objectID or None for FP or -1 for ignored area)

            # For each objectID, get the list of its corresponding gt_objectID, i.e. list of int/None/-1
            for idx, gt_objectID in dict_gt_objectID_frame.items():
                objectID = obsColl[idx].objectID
                if objectID in dict_gt_objectID_tracklets:
                    dict_gt_objectID_tracklets[objectID] += [gt_objectID]
                else:
                    dict_gt_objectID_tracklets[objectID] = [gt_objectID]

        # Return the most frequent gt_object per objectID
        # NB : -1 are firstly ignored, meaning for [42, 42, -1, -1, -1], (2 observations TP as object 42 and 3 observations in ignored area), returns 42
        dict_gt_objectID_tracklets_no_ignored = {objectID: [gt_objectID for gt_objectID in list_gt_objectID if gt_objectID != -1]
                                                 for objectID, list_gt_objectID in dict_gt_objectID_tracklets.items()
                                                 if not all(v == -1 for v in list_gt_objectID)}  # ignore areas that are ignored

        most_frequent_gt_objectID = {objectID: Counter(list_gt_objectIDs).most_common(1)[0][0] for objectID, list_gt_objectIDs in dict_gt_objectID_tracklets_no_ignored.items()}
        # NB : scipy.stats.mode() does not work well when the mode is 'None'

        # Add -1, i.e. 'ignored', for objectID which are absent from most_frequent_gt_objectID
        for objectID in dict_gt_objectID_tracklets.keys():  # all objectID in the tracklets
            if objectID not in most_frequent_gt_objectID:
                most_frequent_gt_objectID[objectID] = -1

        return most_frequent_gt_objectID  # {objectID: int if TP, -1 if ignored, None if FP}

    def to_box(self):

        for f, obsCollection in self.tracklets.items():
            obsCollection.to_box()

    def to_mask(self):

        for f, obsCollection in self.tracklets.items():
            obsCollection.to_mask(img_width=self.scene.width, img_height=self.scene.height)

    def merge_same_objectIDs(self, **kwargs):
        """
        Replacement the objectIDs by a common one
        """

        self.all_tracks = self.get_all_tracks()
        for f, obsCollection in self.tracklets.items():
            obsCollection.merge_same_objectID(**kwargs)

        self.all_tracks = self.get_all_tracks()

    def __getitem__(self, frame: int) -> ObsCollection:
        return self.tracklets.get(frame)

    def __len__(self) -> int:
        return len(self.tracklets)

    def __str__(self):
        text = f"There are {len(self.all_tracks)} tracks in these {len(self)} frames."
        for f, obsCollection in self.tracklets.items():
            text += f"\n\nAt the frame {f:03d}, {obsCollection}"
        return text

    def save(self, folder, viz=True, benchmark=True, level=None, **kwargs):
        """
        Save the results of a tracker
        :param folder: path to the saving
        :param viz: visualize the tracker
        :param benchmark: save the tracker in the format of the challenge
        :param level: level of precision of locators
        """

        if level == 'box':  # Replace masks by box in 2D MOT
            self.to_box()

        if benchmark:
            self.scene.save_benchmark(self.tracklets, folder=folder, **kwargs)

        if viz:
            self.visualize(path_save=Path(folder) / f"{self.scene.scene_name}.avi", n_colors=10, show_boxes=level == 'box')
