# Results

This folder contains the outputs of detectors and trackers.

## Download detections

Run the following code to download and unzip the detections on MOT17, DanceTrack and KITTIMOT:
```bash
wget https://mehdimiah.com/static/twix/results.zip
unzip results.zip
rm results.zip
```

On MOT17 and DanceTrack, the detections were generated using the weights of [ByteTrack](https://github.com/ifzhang/ByteTrack) and on KITTIMOT, the detections were those of [Permatrack](https://github.com/TRI-ML/permatrack/issues/16).

## File organization

The results of tracking will be stored in the folder `results/<dataset>-<subset>/Tracking/`.

The files are organized as follows:
```plaintext
results
├── DanceTrack-test
│   ├── Detection
|   |   └── bytetrack_model
|   |       ├── dancetrack0003.txt
|   |       ├── dancetrack0009.txt
|   |       └── ...
|   └── Tracking
|       └── ...
└── ...
```

## Format of detection files

All detection files of 2D boxes follow the same format: 
```plaintext
<frame: int> <objectID: int> <classe_id: int> <score: float> <xmin: float> <ymin: float> <xmax: float> <ymax: float>
```

The `frame` is frame name for the supplied images. It usually starts at 0 or 1.\
The `objectID` are all initialized at 0. During the tracking, a strictly positive `objectID` is used to identity tracks.\
The `classe_id` is the index in the [COCO classnames](../src/detection/__init__.py).\
The `score` is the confidence score between 0 and 1.\
The coordinates (`xmin`, `ymin`) is the top-left corner and (`xmax`, `ymax`) the bottom-right corner of a detection.

## Format of tracking files

All tracking files follow the specific format of each dataset.

For MOT17 and DanceTrack, the output format is as follows:
```plaintext
<frame: int>, <objectID: int>, <xmin: float>, <ymin: float>, <width: float>, <height: float>, <score: float>, -1, -1, -1
```

For KITTIMOT, the output format is as follows:
```plaintext
<frame: int> <objectID: int> <classe: Union['Car', 'Pedestrian']> 0 0 0 <xmin: float> <ymin: float> <xmax: float> <ymax: float> 0 0 0 0 0 0 0 <score: float>
```