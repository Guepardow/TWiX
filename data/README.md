# Datasets

We consider that all datasets are stored into a folder named `<PATH_DATA>`.

## MOT17

1) Go to [https://motchallenge.net/data/MOT17/](https://motchallenge.net/data/MOT17/)
2) Download (5.5 GB) and extract the zipped folder `MOT17.zip` in the folder `<PATH_DATA>/MOT17`.

The organization of MOT17 is as follows:

```plaintext
MOT17
├── test
│   ├── MOT17-01-DPM
│   │   ├── det
│   │   ├── img1
│   │   └── seqinfo.ini
│   └──  ...
└── train
    ├── MOT17-02-DPM
    │   ├── det
    │   ├── gt
    │   ├── img1
    │   └── seqinfo.ini
    └──  ...

```

## DanceTrack

1) Go to [https://github.com/DanceTrack/DanceTrack](https://github.com/DanceTrack/DanceTrack)
2) Download the zipped files (total 18 GB) and extract them in the folder `<PATH_DATA>/DanceTrack`.

The organization of DanceTrack is as follows:

```plaintext
DanceTrack
├── test
|   ├── dancetrack0003
|   |   ├── img1
|   |   └── seqinfo.ini
|   └── ...
├── train
|   ├── dancetrack0001
|   |   ├── gt
|   |   ├── img1
|   |   └── seqinfo.ini
|   └── ...
└── val
    ├── dancetrack0004
    |   ├── gt
    |   ├── img1
    |   └── seqinfo.ini
    └── ...
```

## KITTIMOT

1) Go to [https://www.cvlibs.net/datasets/kitti/eval_tracking.php](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)
2) Register to accept the licensing conditions
3) Download (total 15.8 GB) the zipped folders `data_tracking_image_2.zip` and `data_tracking_label_2.zip` and extract them in the folder `<PATH_DATA>/KITTIMOT`.

The organization of KITTIMOT is as follows:

```plaintext
KITTIMOT
├── data_tracking_image_2
|   ├── testing
|   └── training
└── data_tracking_label_2
    └── training
        └── label_02
            ├── 0000.txt
            ├── 0001.txt
            └── ...
```

## MOT20

1) Go to [https://motchallenge.net/data/MOT20/](https://motchallenge.net/data/MOT20/)
2) Download (5.0 GB) and extract the zipped folder `MOT20.zip` in the folder `<PATH_DATA>/MOT20`.

The organization of MOT20 is as follows:

```plaintext
MOT20
├── test
│   ├── MOT20-04
│   │   ├── det
│   │   ├── img1
|   |   └── seqinfo.ini
│   └── ...
└── train
    ├── MOT20-01
    │   ├── det
    │   ├── gt
    │   ├── img1
    |   └── seqinfo.ini
    └── ...

```