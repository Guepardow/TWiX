#!/bin/sh

# MOT17 - Elapsed time: 2h30

# Creation of detection
# Download them following the instructions in the results/README.md

# Creation of tracklets using TrackerIoU
cd ../src/tracker
for subset in train_half val_half ; do python trackerIoU.py --dataset MOT17 --subset $subset --detection bytetrack_x_mot17 ; done ;

# Creation of batches of data for short (STA) and long-term association (LTA)
cd ../association/twix
for subset in train_half val_half ; do python data.py --dataset MOT17 --subset $subset --detection bytetrack_x_mot17 --WP 0.4s --WF 1f --strategy frame --max_gap 0.0 ; done ;
for subset in train_half val_half ; do python data.py --dataset MOT17 --subset $subset --detection bytetrack_x_mot17 --WP 0.4s --WF 1f --strategy tracklet --max_gap 0.8 ; done ;

# Training of the TWiX modules at the STA and LTA levels
python train.py --dataset MOT17 --subset_train train_half --subset_val val_half --WP 0.4s --WF 1f --strategy frame --max_gap 0.0 --num_layers 1 --lr 0.0001 --inter_pair
python train.py --dataset MOT17 --subset_train train_half --subset_val val_half --WP 0.4s --WF 1f --strategy tracklet --max_gap 0.8 --num_layers 4 --lr 0.001 --inter_pair

# Evaluation of the C-TWiX tracker
cd ../../tracker
python c-twix.py --dataset MOT17 --subset val_half --detection bytetrack_x_mot17 --min_score 0.50 --min_area 128 --method_twix_1 twix_sta_mot17_half --theta_1 0.8 --method_twix_2 twix_lta_mot17_half --theta_2 -0.4 --max_age 0.8 --min_score_new 0.70
