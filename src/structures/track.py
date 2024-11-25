import numpy as np

from .locator import Locator
from .observation import Observation


class Track:
    """
    Collection of observations related to the SAME OBJECT
    This is a special case of ObsCollection where: 
    - the objectID are all the same
    - the classe are all the same
    - there is no duplicates in frame
    """

    def __init__(self, objectID: int, classe: str, locator: Locator, frame: int, score: float):

        self.locators = [locator]                           # Will contain all locators
        self.coordinates = np.array([locator.coordinates])  # Will contain all coordinates
        self.frames = np.array([frame])                     # Will contain all frames
        self.scores = np.array([score])                     # Will contain all scores

        self.state = np.array([locator.coordinates])        # Last state (known or estimated) during online tracking
        self.age = 0                                        # Age of the track since its last appearition (in seconds)

        # These properties will not change
        self.objectID = objectID
        self.classe = classe

    def add_observation(self, locator: Locator, frame: int, score: float):

        assert frame not in self.frames, f"Frames duplicates detected: frame {frame} is in {self.frames}"

        self.locators = self.locators + [locator]

        self.coordinates = np.vstack((self.coordinates, locator.coordinates))
        self.frames = np.hstack((self.frames, frame))
        self.scores = np.hstack((self.scores, score))
        self.age = 0  # since this object is observed, we reset the age

        self.update_state(new_state = locator.coordinates)

    def update_state(self, new_state):
        self.state = new_state

    def __getitem__(self, item) -> Observation:

        return Observation(classe=self.classe, score=self.scores[item], frame=self.frames[item],
                           locator=self.locators[item], objectID=self.objectID, flag='_X_') 

    def __iter__(self):
        return ObsIterator(self)

    def __len__(self):
        return len(self.frames)
    

class ObsIterator:
    def __init__(self, obs):
        self._obs = obs
        self._index = 0

    def __next__(self):
        if self._index < (len(self._obs)):
            result = self._obs[self._index]
            self._index += 1
            return result
        raise StopIteration