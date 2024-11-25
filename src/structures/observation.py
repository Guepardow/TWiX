from __future__ import annotations
from typing import List, Optional

from .locator import Locator


class Observation:
    """
    A class used to fully describe an observation
    """

    def __init__(self, classe: Optional[str], score: Optional[float], frame: int,
                 locator: Locator, objectID: int, flag: str):
        """
        :param classe: object class name
        :param score: confidence score for this observation (preferably between 0 and 1)
        :param frame: frame number for this observation
        :param locator: locator (e.g. Point, Box, Mask) for this observation
        :param objectID: identifier of the object of interest. Value at 0 indicates a detection without any association
        :param flag: a comment for this observation (e.g. '_DET_', '_GT_', '_NSO_', '_MERGE_', '_LI_', '_IGN_', '_X_' for respectively
                indicating a detection, a ground truth, a modified Mask by non spatial overlap algorithm, a merge between
                multiple observations, a generated observation by linear interpolation, ignored observations and unknown.

        Examples:
        ---------
        >>> obs = Observation(classe='person', score=0.85, frame=1, locator=Mask(rle=r"XQa53c;4L2O001O0000O2N2NYTb8", img_width=1242, img_height=375), objectID=1, flag='_GT_')

        For an ignored region, classe and score are at None
        """

        assert (classe is None) or isinstance(classe, str), f"The type of classe {classe} should be <str> and not {type(classe)}"
        assert objectID >= 0, f"ObjectID should be greater than 0: {objectID}"
        assert flag in ('_DET_', '_GT_', '_NSO_', '_MERGE_', '_LI_', '_IGN_', '_X_'), f"flag {flag} is not canon"

        self.objectID = objectID
        self.classe = classe
        self.score = score
        self.frame = frame
        self.locator = locator
        self.flag = flag

    def merge(self, others: List[Observation]) -> Observation:
        """
        Fusion two observations of the same object.
        The two objects must have the same objectID, frame and class ; and their locator must be from the same class
        """

        for other in others:
            assert self.objectID == other.objectID, f"Not same objectID: {self.objectID} vs {other.objectID}"
            assert self.classe == other.classe, f"Not same classe: {self.classe} vs {other.classe}"
            assert self.frame == other.frame, f"Not same frame: {self.frame} vs {other.frame}"
            assert self.locator.__class__ == other.locator.__class__, f"Not same locator's class : {self.locator.__class__} vs {other.locator.__class__}"

        # New score : the maximum
        score = max([self.score] + [other.score for other in others])

        # New locator : use the built-in merge method
        locator = self.locator.merge([other.locator for other in others])

        return Observation(self.classe, score, self.frame, locator, self.objectID, flag='_MERGE_')

    def __str__(self):
        text = f"This observation ({self.flag}) of the classe {self.classe} {self.objectID} " \
               f"is detected with a score of {self.score} " \
               f"at the frame {self.frame} @{self.locator.coordinates} by a {self.locator.__class__.__name__}."
        return text
