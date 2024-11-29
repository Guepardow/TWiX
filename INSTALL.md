# Installation

Tested on hardware: Nvidia RTX 2060 (6 GB VRAM), AMD Ryzen 5 3600X (6 cores), 16 GB RAM\
Tested on system: Fedora Linux 41, GCC 14.2.1, CUDA 12.4, Python 3.10.12, PyTorch 2.4.1

## Environment

Clone this repo and install packages:

```bash
git clone https://github.com/Guepardow/TWiX
cd TWiX

conda create -n twix python=3.10.12
conda activate twix
conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install numpy==2.1.3 pandas==2.2.2 matplotlib==3.8.4 seaborn==0.13.2 opencv-python==4.10.0.84 einops==0.8.0 pycocotools==2.0.8 tensorboard==2.18.0 scipy==1.14.1 tqdm prettytable notebook loguru pyyaml
```

## Evaluation with TrackEval

For MOT17, MOT20 and KITTIMOT, the evaluation is done using [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA metrics).
[DanceTrack](https://github.com/DanceTrack/DanceTrack) is not officially supported by TrackEval, that is why it is necessary to download it separately.

### Installation of TrackEval

```bash
# Clone official TrackEval and for DanceTrack
cd src/evaluation
git clone https://github.com/JonathonLuiten/TrackEval.git
git clone https://github.com/DanceTrack/DanceTrack.git

# Download the data with the ground truth (150 MB)
cd TrackEval
wget https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip
unzip -q data.zip ; rm data.zip
```

### Patches on TrackEval and DanceTrack

After downloading TrackEval and DanceTrack, some script modifications are necessary (<https://github.com/JonathonLuiten/TrackEval/pull/50>):

```bash
# on scripts
sed -i "75i\            elif setting in ['SEQMAP_FOLDER', 'OUTPUT_FOLDER']:" scripts/run_mot_challenge.py ../DanceTrack/TrackEval/scripts/run_mot_challenge.py
sed -i "76i\                x = args[setting][0]" scripts/run_mot_challenge.py ../DanceTrack/TrackEval/scripts/run_mot_challenge.py
```

Additionally, occurrences of `np.float` and `np.int` were replaced with `float` and `int`, respectively (<https://github.com/JonathonLuiten/TrackEval/pull/117>):

 ```bash
# on datasets
sed -i 's/np\.float/float/g' trackeval/datasets/mot_challenge_2d_box.py trackeval/datasets/kitti_2d_box.py ../DanceTrack/TrackEval/trackeval/datasets/mot_challenge_2d_box.py
sed -i 's/np\.int/int/g' trackeval/datasets/mot_challenge_2d_box.py trackeval/datasets/kitti_2d_box.py ../DanceTrack/TrackEval/trackeval/datasets/mot_challenge_2d_box.py

# on metrics
sed -i 's/np\.float/float/g' trackeval/metrics/hota.py ../DanceTrack/TrackEval/trackeval/metrics/hota.py
sed -i 's/np\.int/int/g' trackeval/metrics/identity.py ../DanceTrack/TrackEval/trackeval/metrics/identity.py
```

### Creation of validation sets for MOT17 and MOT20

Since MOT17 and MOT20 do not have any validation set in TrackEval, we create them:

```bash

# Create halves train_half and val_half on MOT17 and MOT20 if needed
cd ../../../tools
python create_halves_mot17.py -p ../src/evaluation/TrackEval/
python create_halves_mot20.py -p ../src/evaluation/TrackEval/
```
