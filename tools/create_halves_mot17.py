import argparse
import configparser
import pandas as pd
from pathlib import Path
from loguru import logger


LIST_SCENES = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
SPLITTER_HALF = {'train_half': lambda x: x[:len(x)//2], 'val_half': lambda x: x[len(x)//2:]}

# Create a custom ConfigParser class to preserve capitalization
class MyConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr


def get_parser():
    parser = argparse.ArgumentParser(description="Create halves on MOT17")
    parser.add_argument('--path_trackeval', '-p', type=str, help="Path to TrackEval")
    return parser


def main(opts):
        
    for subset in ['train_half', 'val_half']:

        # Create a seqmap file
        pd.DataFrame({'name': LIST_SCENES}).to_csv(Path(f"{opts.path_trackeval}/data/gt/mot_challenge/seqmaps/MOT17-{subset}.txt"), index=False)
                                        
        for scene in LIST_SCENES:
            
            config_train = MyConfigParser()
            config_half = MyConfigParser()
            
            # Create a folder on the half set
            Path(f"{opts.path_trackeval}/data/gt/mot_challenge/MOT17-{subset}/{scene}").mkdir(parents=True, exist_ok=True)

            # Read the seqinfo from the train and create a seqinfo.ini on the half
            config_train.read(Path(f"{opts.path_trackeval}/data/gt/mot_challenge/MOT17-train/{scene}-DPM/seqinfo.ini"))
            for section in config_train.sections():
                config_half.add_section(section)
                for option in config_train.options(section):
                    value = config_train.get(section, option)

                    if option == 'name':
                        # remove detections name : no need it for private setting with my own detection
                        config_half.set(section, option, '-'.join(value.split('-')[:2]))
                    else:
                        config_half.set(section, option, value)
                    # seqLength remains unchanged ! 

            # Save the config as a seqinfo for each scene
            with open(Path(f"{opts.path_trackeval}/data/gt/mot_challenge/MOT17-{subset}/{scene}/seqinfo.ini"), 'w') as configfile:
                config_half.write(configfile, space_around_delimiters=False)

            # Create a gt file on the half
            Path(f"{opts.path_trackeval}/data/gt/mot_challenge/MOT17-{subset}/{scene}/gt").mkdir(parents=True, exist_ok=True)
            
            gt_train = pd.read_csv(Path(f"{opts.path_trackeval}/data/gt/mot_challenge/MOT17-train/{scene}-DPM/gt/gt.txt"), sep=',', header=None)
            list_frames = sorted(set(gt_train[0]))
            list_frames_half = SPLITTER_HALF[subset](list_frames)

            gt_half = gt_train.loc[gt_train[0].isin(list_frames_half)]
            gt_half.to_csv(Path(f"{opts.path_trackeval}/data/gt/mot_challenge/MOT17-{subset}/{scene}/gt/gt.txt"), sep=',', header=None, index=False)
            
        logger.success(f"Successfully created seqmaps for all scenes on MOT17-{subset} set")  

if __name__ == '__main__':

    args = get_parser().parse_args()
    main(args)