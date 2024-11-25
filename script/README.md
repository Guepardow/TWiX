# Script

For each dataset, the following scripts:
1. create the tracklets using TrackerIoU on the training and validation sets;
2. create batches of data for short and long-term association on the training and validation sets;
3. train the TWiX module on the training data of batches for short and long-term association;
4. run the tracker C-TWiX on the validation set.

Run the following code to execute the script on MOT17:
```bash
bash c-twix_MOT17.sh
```

Run the following code to execute the script on DanceTrack:
```bash
bash c-twix_DT.sh
```

Run the following code to execute the script on KITTIMOT:
```bash
bash c-twix_KT.sh
```