<p align="center">
    <img src="assets/logo_twix.png" class="center" width="300"/> <br>
    <a href="https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=learning-data-association-for-multi-object"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-data-association-for-multi-object/multi-object-tracking-on-mot17" alt="PWC" height="18" /></a> <br>
    <a href="https://paperswithcode.com/sota/multiple-object-tracking-on-kitti-test-online?p=learning-data-association-for-multi-object"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-data-association-for-multi-object/multiple-object-tracking-on-kitti-test-online" alt="PWC" height="18" /></a> <br>
    <a href="https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=learning-data-association-for-multi-object"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-data-association-for-multi-object/multi-object-tracking-on-dancetrack" alt="PWC" height="18" /></a> <br>
</p>

> **Learning Data Association for Multi-Object Tracking using Only Coordinates**
>
> *Mehdi Miah, Guillaume-Alexandre Bilodeau, Nicolas Saunier*
>
> [arXiv 2403.08018](https://arxiv.org/abs/2403.08018), [Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0031320324009208)
>

## TL;DR

- C-TWiX is an **online tracker** (tracking-by-detection) that uses **only coordinates** (no reID, no 3D, no CMC);
- C-TWiX relies on the **cascade matching** of [C-BIoU](https://arxiv.org/abs/2211.14317) and replaces the heuristic BIoU by our TWiX module;
- Given $M$ tracks and $N$ detections, the TWiX module returns a $M \times N$ similarity **matrix**;
- Similarity for each track-detection pair is computed by considering the positions of **all other objects**;
- Two TWiX modules are trained for **short-term** (on adjacent frames) and **long-term** association;
- C-TWiX achieves **SOTA performance** on DanceTrack and KITTIMOT datasets, and is competitive on MOT17.

<p align="center"><img src="assets/animation.gif" class="center" alt="animation" width="600"/></p>

## Abstract

We propose a novel Transformer-based module to address the data association problem for multi-object tracking. From detections obtained by a pretrained detector, this module uses only coordinates from bounding boxes to estimate an affinity score between pairs of tracks extracted from two distinct temporal windows. This module, named TWiX, is trained on sets of tracks with the objective of discriminating pairs of tracks coming from the same object from those which are not. Our module does not use the intersection over union measure, nor does it requires any motion priors or any camera motion compensation technique. By inserting TWiX within an online cascade matching pipeline, our tracker C-TWiX achieves state-of-the-art performance on the DanceTrack and KITTIMOT datasets, and gets competitive results on the MOT17 dataset.

## Installation

1. Follow the instructions in [INSTALL.md](INSTALL.md) to install the environment and download additional repos
2. Follow the instructions in [data/README.md](data/README.md) to download the MOT17, DanceTrack and KITTIMOT datasets
3. For downloading detections on these datasets, refer to [results/README.md](results/README.md)

## Training

To train a TWiX module, please follow these steps:

```bash
# 1) run a detector on each frame of a video, or use the detections provided in ./results/*/Detection
# code not provided

# 2) apply the IoU-Tracker on the video to create tracklets on the training and validation sets
cd src/tracker
for subset in train val ; do python trackerIoU.py --dataset DanceTrack --subset $subset --detection bytetrack_model ; done ;

# 3) create batches of data on the training and validation sets
cd ../association/twix
for subset in train val ; do python data.py --dataset DanceTrack --subset $subset --detection bytetrack_model --WP 0.8s --WF 1f --strategy frame --max_gap 0.0 ; done ; 

# 4) train the TWiX on these batches of data
python train.py --dataset DanceTrack --subset_train train --subset_val val --WP 0.8s --WF 1f --strategy frame --max_gap 0.0 --num_layers 1 --lr 0.0001 --inter_pair
```

For dataset specific code, please refer to scripts in the folder [script](script).

## Inference

To run the tracker C-TWiX on a video, please follow these steps:

```bash
# 1) run a detector on each frame of a video, or use the detections provided in ./results/*/Detection
# code not provided

# 2) run the tracker C-TWiX
cd src/tracker
python c-twix.py --dataset DanceTrack --subset val --detection bytetrack_model --min_score 0.50 --min_area 128 --method_twix_1 <name_of_twix_1_exp> --theta_1 -0.4 --method_twix_2 <name_of_twix_2_exp> --theta_2 -0.2 --max_age 1.6 --min_score_new 0.90
```

For dataset specific code, please refer to scripts in the folder [script](script).

## Results

### On the official test sets

|    C-TWiX                                                                                | HOTA | DetA | AssA | MOTA | IDF1 |  Speed |
|------------------------------------------------------------------------------------------|------|------|------|------|------|--------|
| [DanceTrack](https://codalab.lisn.upsaclay.fr/competitions/5830#results)                 | 62.1 | 81.8 | 47.2 | 91.4 | 63.6 | 300 Hz |
| [MOT17](https://motchallenge.net/results/MOT17/?det=Private&orderBy=HOTA&orderStyle=ASC) | 63.1 | 64.1 | 62.5 | 78.1 | 76.3 |  50 Hz |
| [KITTIMOT-car](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)                  | 77.6 | 77.0 | 78.8 | 89.7 |  NA  | 320 Hz |
| [KITTIMOT-ped](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)                  | 52.4 | 50.8 | 54.4 | 65.0 |  NA  | 320 Hz |

To download TWiX weights, follow the instructions in [src/association/twix/README.md](src/association/twix/README.md).

### On the validation sets

This table contains the HOTA score on the validation sets of some trackers:

- **[TrackerIoU](src/tracker/trackerIoU.py)** : this tracker associates detections from adjacent frames if the IoU is higher than a threshold in a single matching pipeline;
- **[S-TWiX](src/tracker/s-twix.py)**: this tracker associates using one TWiX module in a single matching pipeline;
- **[C-TWiX](src/tracker/c-twix.py)**: this tracker associates using two TWiX modules in a cascade matching pipeline;
- **[oracleLTA](src/tracker/oracleLTA.py)**: given the tracklets returned by TrackerIoU, this tracker associates perfectly the tracklets that correspond to the same ground truth object;
- **[oracleAsso](src/tracker/oracleAsso.py)**: this tracker associates perfectly the detections to their groud truth object if they are true positives and removes any false positives.

|   Dataset    | Detections                                                   | TrackerIoU | S-TWiX | C-TWiX | oracleLTA | oracleAsso |
|--------------|--------------------------------------------------------------|------------|--------|--------|-----------|------------|
| MOT17        | [YOLOX from ByteTrack](https://github.com/ifzhang/ByteTrack) | 71.8       | 77.1   |  77.8  | 81.0      | 83.7       |
| DanceTrack   | [YOLOX from ByteTrack](https://github.com/ifzhang/ByteTrack) | 44.7       | 58.6   |  60.4  | 74.6      | 84.3       |
| KITTIMOT-car | [Permatrack](https://github.com/TRI-ML/permatrack/issues/16) | 84.5       | 88.4   |  89.3  | 88.6      | 90.5       |
| KITTIMOT-ped | [Permatrack](https://github.com/TRI-ML/permatrack/issues/16) | 63.9       | 70.2   |  71.4  | 70.9      | 73.9       |

The validations sets are obtained as follows:

- the validation set of DanceTrack is the [official set](https://github.com/DanceTrack/DanceTrack);
- the validation set of MOT17 is obtained by splitting the video in half, following [Zhou et al](https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/convert_mot_to_coco.py);
- the validation set of KITTIMOT is obtained using the split of KITTIMOTS, following [Luiten et al](https://github.com/JonathonLuiten/TrackEval/tree/master).

## Citation and acknowledgement

If you refer to this work, please cite :

```bibtex
@article{miah2024learningdata,
title = {Learning data association for multi-object tracking using only coordinates},
journal = {Pattern Recognition},
volume = {160},
pages = {111169},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.111169},
url = {https://www.sciencedirect.com/science/article/pii/S0031320324009208},
author = {Mehdi Miah and Guillaume-Alexandre Bilodeau and Nicolas Saunier},
keywords = {Tracking, Transformer, Data association, Motion, Multi-object tracking}
}
```

We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC) [funding reference number RGPIN-2020-04633 and RGPIN-2017-06115].