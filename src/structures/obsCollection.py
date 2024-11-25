from __future__ import annotations
from typing import Dict, List, Optional

import prettytable
import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment

from .observation import Observation
from .locator import Point, Box, Mask


class ObsCollection:
    """
    A class used to defined a collection of Observations.

    These observations may belong to :
    - the same frame (e.g. all detections at frame 1) or ;
    - the same object (e.g. all detections of the pedestrian 1) ;
    - a history of tracklets (e.g. all last positions of detected objects)
    """

    def __init__(self):
        self.observations = []  # All observations

    def add_observation(self, observation: Observation):
        self.observations.append(observation)

    def remove_idx(self, idx: int):
        """
        Removes an element from the collection of observation.
        NB: this function is useful when the objectIDs are not initialized
        """
        self.observations.pop(idx)

    def remove_objectID(self, objectID: int):
        """
        Removes all observations identified by the objectID
        """
        self.observations = [obs for obs in self.observations if obs.objectID != objectID]

    def keep_class(self, classes_to_keep: List[str]):
        """
        Removes all observations not identified by the classes_to_keep
        """
        self.observations = [obs for obs in self.observations if obs.classe in classes_to_keep]

    def keep_big_objects(self, min_area: float):
        """
        Removes all observations with an area smaller than min_area
        """
        self.observations = [obs for obs in self.observations if obs.locator.area >= min_area]

    def keep_high_score(self, min_score: float):
        """
        Removes all observations with a score lower than min_score
        """
        self.observations = [obs for obs in self.observations if obs.score >= min_score]

    def is_not_initialized(self) -> bool:
        return all([obs.objectID == 0 for obs in self])

    def mode_class(self) -> str:
        """
        For a track, return the class with the highest accumulation of scores.
        """
        list_classes = list(set([obs.classe for obs in self.observations]))
        dict_classe = {classe: 0 for classe in list_classes}
        for obs in self.observations:
            dict_classe[obs.classe] += obs.score

        return max(dict_classe, key=dict_classe.get)

    def is_complete(self, framestep: int) -> bool:
        """
        Returns True if the collection of Observation is a temporally complete sequence
        NB: this is useful to know if a tracklet (as a collection of Observation) is complete

        :param framestep: step in the frames of the scene
        """

        assert len(set([obs.frame for obs in self])) == len(self), f"The current collection of observations has some observations coming from the same frame : {self}"
        assert len(set([obs.objectID for obs in self])) == 1, f"The current collection of observations has some observations from multiple objects: {self}"

        first_frame_obs, last_frame_obs = min([obs.frame for obs in self]), max([obs.frame for obs in self])

        return len(self) == ((last_frame_obs - first_frame_obs) / framestep + 1)

    def to_box(self):

        for obs in self.observations:
            obs.locator = obs.locator.to_box()

    def to_mask(self, img_width: int, img_height: int):

        for obs in self.observations:
            obs.locator = obs.locator.to_mask(img_width, img_height)

    def get_mask_ignored(self, img_width: int, img_height: int) -> Mask:
        """
        Transform a collection of _IGN_ observations into a single Mask
        :param img_width: width of the image
        :param img_height: height of the image

        NB: usually, if the dataset is about MOTS, the ignored region is a single Mask
                   , if the dataset is about MOT, the ignored region is a collection of Box
        """

        assert all([obs.flag == '_IGN_' for obs in self.observations]), "Some observations are not ignored area"

        bin_map_ignored = np.zeros((img_height, img_width), dtype=bool)
        for obs in self.observations:
            bin_map_ignored += obs.locator.to_binmap(img_width=img_width, img_height=img_height)
        mask_ignored = Mask.from_binmap(tl=Point(0, 0), bin_map=bin_map_ignored, img_width=img_width, img_height=img_height)

        return mask_ignored

    def get_gt_frame(self, gt_obsColl: ObsCollection, mask_ignored: Mask, threshold: float, img_width: int, img_height: int) -> Dict[int, Optional[int]]:
        """
        Returns the association between an observation and a true detection.
        If it is a true positive, returns the object_ID of the ground truth object
        Otherwise, if it falls in an ignored region, return -1
        Otherwise, it is a false positive and we return None

        :param gt_obsColl: collection of ground truth observations
        :param mask_ignored: a Mask with the ignored regions
        :param threshold: minimal IoU/mIoU to get to be considered as a true positive
        :param img_width: width of the image
        :param img_height: height of the image
        """

        assert '_IGN_' not in [obs.flag for obs in gt_obsColl]
        assert isinstance(mask_ignored, Mask), "Ignored regions are not embedded in a Mask"
        assert len(set([obs.frame for obs in self])) <= 1, f"Observations are coming from multiple frames {self}"
        # NB : the length is 0 when the obsCollection is empty, or 1 if it contains some observations
        
        cost = np.zeros((len(self), len(gt_obsColl)))
        for i, obs in enumerate(self.observations):
            for j, gt_obs in enumerate(gt_obsColl):
                if isinstance(gt_obs.locator, Box):
                    cost[i, j] = 1-obs.locator.IoU(gt_obs.locator)
                elif isinstance(gt_obs.locator, Mask):
                    cost[i, j] = 1-obs.locator.mIoU(gt_obs.locator)

        row_ind, col_ind = linear_sum_assignment(cost)  # Hungarian algorithm
        # NB: using the Hungarian algorithm prevents double associations, contrary to a naive 50% overlap

        dict_gt_objectID = {i: None for i, _ in enumerate(self)}  # all observations are initialized as false positive
        for i, j in zip(row_ind, col_ind):
            if 1-cost[i, j] > threshold:  # it is a correct match
                dict_gt_objectID[i] = gt_obsColl[j].objectID  # attribute the objectID of its corresponding GT

        # Here, values of dict_gt_objectID that are equal to None indicates either a false positive or an observation in an ignored region
        for i, gt_objectID in dict_gt_objectID.items():
            if gt_objectID is None:  # still considered as a false positive, because it does not match any gt observation
                if self.observations[i].locator.to_mask(img_width=img_width, img_height=img_height).mIoM(mask_ignored) > threshold:
                    dict_gt_objectID[i] = -1

        return dict_gt_objectID  # {0: 45, 1: 51, 2: None, 3: 46, 4: -1} (position: gt_objectID or None if FP or -1 if ignored)

    def merge_same_objectID(self, **kwargs):

        # Count the number of same objects for each object
        counter_objectID = Counter([obs.objectID for obs in self.observations])

        # For each object appearing more than once, a fusion is applied
        for objectID, coeff in counter_objectID.items():
            if coeff > 1:

                # Position of the same object
                idx_same = [i for i, ID in enumerate([obs.objectID for obs in self.observations]) if ID == objectID]

                # The fusion follow these rules:
                # same objectID, dominant class, maximum score, dominant frames, merged locator

                classe_same = [self.observations[idx].classe for idx in idx_same]
                classe_mode = max(set(classe_same), key=classe_same.count)

                frame_same = [self.observations[idx].frame for idx in idx_same]
                frame_mode = max(set(frame_same), key=frame_same.count)

                score_same = [self.observations[idx].score for idx in idx_same]
                score_max = np.max(score_same)

                locator_same = [self.observations[idx].locator for idx in idx_same]
                locator_merge = locator_same[0].merge(locator_same[1:], **kwargs)

                self.remove_objectID(objectID)

                new_observation = Observation(classe=classe_mode, score=score_max, frame=frame_mode, locator=locator_merge,
                                              objectID=objectID, flag='_MERGE_')
                self.add_observation(new_observation)

    def __getitem__(self, item: int) -> Observation:
        return self.observations[item]

    def __iter__(self):
        return ObsIterator(self)

    def __len__(self):
        return len(self.observations)

    def __str__(self):
        if len(self) == 0:
            return 'There is no object at the moment'
        elif len(self) == 1:
            text = "There is 1 observation."
        else:
            text = f"There are {len(self)} observations."

        if len(self) != 0:
            lf_table = prettytable.PrettyTable(['#', 'objectID', 'classe', 'score', 'frame', 'type', 'coordinates', 'center', 'area', 'flag'], hrules=prettytable.HEADER)
            for idx, obs in enumerate(self.observations):
                lf_table.add_row([idx, obs.objectID, obs.classe, round(obs.score, 3), obs.frame, obs.locator.__class__.__name__, 
                [int(value) for value in obs.locator.coordinates], [int(value) for value in obs.locator.center.coordinates], obs.locator.area, obs.flag])

            return f"{text}\n{lf_table.get_string()}"
        

class ObsIterator:
    def __init__(self, obsCollection: ObsCollection):
        self._obsCollection = obsCollection
        self._index = 0

    def __next__(self):
        if self._index < (len(self._obsCollection)):
            result = self._obsCollection[self._index]
            self._index += 1
            return result
        raise StopIteration
