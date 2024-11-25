# Source code

This folder contains the core components required to run a tracking-by-detection tracker. Such trackers use detections provided by pre-trained object detection models, enabling them to leverage advancements in object detection for improved tracking performance.

In a tracking-by-detection pipeline, the following steps are crucial:
- __detection step__: defines how objects are detected in images;
- __association step__: specifies how detections are linked together to form tracks;
- __pipeline__: determines how the tracking algorithm operates, including:
    - single vs. cascade matching;
    - handling of false positives;
    - recovery from false negatives.

## File organization

This folder is organized as follows:
- `association`: contains algorithms for the __association step__, such as IoU-based methods and TWiX;
- `datasets`: contains the information related to datasets such as the data location, the access to frame image, the framerate, the methods to save tracking results and to evaluate the performance of a tracker, etc. Currently, this folder contains the dataset classes MOT17, DanceTrack, KITTIMOT and MOT20;
- `detection`: contains COCO classes and provides space to integrate custom __detector code__;
- `evaluation`: includes the official evaluation scripts. Refer to [INSTALL.md](../INSTALL.md) for setup instructions;
- `structures`: contains foundational classes needed to implement a tracker;
- `tracker`: contains the tracking algorithm, defining the __pipeline__ used for tracking;
- `viz`: provides code for generating visualizations, such as videos showcasing ground truth annotations, detections, and tracking results.