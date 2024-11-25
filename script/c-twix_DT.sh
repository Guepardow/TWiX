#!/bin/sh

# DanceTrack - Elapsed time: 9h

# Creation of detection
# Download them following the instructions in the results/README.md

# Creation of tracklets using TrackerIoU
cd ../src/tracker
for subset in train val ; do python trackerIoU.py --dataset DanceTrack --subset $subset --detection bytetrack_model ; done ;

# Creation of batches of data for short (STA) and long-term association (LTA)
cd ../association/twix
for subset in train val ; do python data.py --dataset DanceTrack --subset $subset --detection bytetrack_model --WP 0.8s --WF 1f --strategy frame --max_gap 0.0 ; done ; 
for subset in train val ; do python data.py --dataset DanceTrack --subset $subset --detection bytetrack_model --WP 0.8s --WF 1f --strategy tracklet --max_gap 1.6 ; done ;

# Training of the TWiX modules at the STA and LTA levels
python train.py --dataset DanceTrack --subset_train train --subset_val val --WP 0.8s --WF 1f --strategy frame --max_gap 0.0 --num_layers 1 --lr 0.0001 --inter_pair
python train.py --dataset DanceTrack --subset_train train --subset_val val --WP 0.8s --WF 1f --strategy tracklet --max_gap 1.6 --num_layers 4 --lr 0.001 --inter_pair

# Evaluation of the C-TWiX tracker
cd ../../tracker
python c-twix.py --dataset DanceTrack --subset val --detection bytetrack_model --min_score 0.50 --min_area 128 --method_twix_1 twix_sta_dancetrack --theta_1 -0.4 --method_twix_2 twix_lta_dancetrack --theta_2 -0.2 --max_age 1.6 --min_score_new 0.90

# Evaluation of the C-TWiX tracker on the GT detections
python c-twix.py --dataset DanceTrack --subset val --detection GT --min_score 0 --min_area 0 --method_twix_1 twix_sta_dancetrack --theta_1 -0.4 --method_twix_2 twix_lta_dancetrack --theta_2 -0.2 --max_age 1.6 --min_score_new 0.90