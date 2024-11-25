#!/bin/sh

# KITTIMOT - Elapsed time: 45 minutes

# Creation of detection
# Download them following the instructions in the results/README.md

# Creation of tracklets using TrackerIoU
cd ../src/tracker
for subset in training_minus_val val ; do python trackerIoU.py --dataset KITTIMOT --subset $subset --detection Permatrack ; done ;

# Creation of batches of data for short (STA) and long-term association (LTA)
cd ../association/twix
for subset in training_minus_val val ; do python data.py --dataset KITTIMOT --subset $subset --detection Permatrack --WP 0.4s --WF 1f --strategy frame --max_gap 0.0 ; done ; 
for subset in training_minus_val val ; do python data.py --dataset KITTIMOT --subset $subset --detection Permatrack --WP 0.4s --WF 1f --strategy tracklet --max_gap 0.8 ; done ;

# Training of the TWiX modules at the STA and LTA levels
python train.py --dataset KITTIMOT --subset_train training_minus_val --subset_val val --WP 0.4s --WF 1f --strategy frame --max_gap 0.0 --num_layers 1 --lr 0.0001 --inter_pair
python train.py --dataset KITTIMOT --subset_train training_minus_val --subset_val val --WP 0.4s --WF 1f --strategy tracklet --max_gap 0.8 --num_layers 4 --lr 0.001 --inter_pair

# Evaluation of the C-TWiX tracker
cd ../../tracker
python c-twix.py --dataset KITTIMOT --subset val --detection Permatrack --min_score 0.50 --min_area 128 --method_twix_1 twix_sta_kittimot_half --theta_1 0.4 --method_twix_2 twix_lta_kittimot_half --theta_2 -0.6 --max_age 0.8 --min_score_new 0.50