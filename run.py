import argparse
import pandas as pd
import yaml

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.load import run_load_data
from src.train import run_training
from src.score import run_scoring


with open('./config/config.yml', "r") as f:
    config = yaml.safe_load(f)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run each section of the model")
    subparsers = parser.add_subparsers()

    # DATA LOADING 
    sb_load = subparsers.add_parser("load", description="Download data")
    sb_load.add_argument('--config', '-c', default=config, help='path to yaml file with configurations')
    sb_load.set_defaults(func=run_load_data)

    # TRAIN 
    sb_train = subparsers.add_parser("train", description="Train model")
    sb_train.add_argument('--model', '-m', default='sim', help="model type: sim or lr")
    sb_train.add_argument('--input', '-i', default='./data/trainSet.csv', help="Path to CSV for training")
    sb_train.set_defaults(func=run_training)    
    
    # SCORE 
    sb_score = subparsers.add_parser("score", description="Score model")
    sb_score.add_argument('--model', '-m', default='knn', help="model type: knn or lr")
    sb_score.add_argument('--input', '-i', default="./data/candidateTestSet.csv", help="Path to CSV for scoring")
    sb_score.add_argument('--output', '-o', default=None, help="Path to CSV for scoring")
    sb_score.set_defaults(func=run_scoring) 
    
    args = parser.parse_args()
    args.func(args)
