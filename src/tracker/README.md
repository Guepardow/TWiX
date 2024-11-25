# Tracker

This folder includes the following trackers:
- __C-TWiX__: an online tracker that follows the tracking-by-detection paradigm. It uses a cascade matching pipeline to associate detections by leveraging two TWiX modules;
- __oracleAsso__: a tracker that uses the ground truth annotation to perfectly associate the detections to their groud truth object if they are true positives and remove any false positives;
- __oracleLTA__: a tracker utilizing ground truth annotations for long-term association. Starting with results from a base tracker (e.g. TrackerIoU), it perfectly associates tracklets belonging to the same ground truth object;
- __S-TWiX__: an online tracker that follows the tracking-by-detection paradigm. It uses a single matching pipeline to associate detections;
- __TrackerIoU__: an online tracker that follows the tracking-by-detection paradigm. It associates detections between consecutive frames based on an IoU threshold, using a single matching pipeline.