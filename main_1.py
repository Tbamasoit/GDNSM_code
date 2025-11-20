import torch
import argparse
import numpy as np # Keep for potential type checks

# 注意：你的 Config 类在 utils/configurator.py 中
from utils.configurator import Config 
from utils.dataset_new import MyDataset # The class we are testing

if __name__ == '__main__':
    # --- 这一块的所有行，都必须用相同数量的空格（推荐4个）来缩进 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of dataset, e.g., baby, sports, tiktok')
    args = parser.parse_args()

    # --- Configuration Loading ---
    print(f"Loading configuration for model 'GDNSM' and dataset '{args.dataset}'...")
    config = Config(model='GDNSM', dataset=args.dataset)
    print("Configuration loaded.")

    test_dataset = MyDataset(config)