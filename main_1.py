import torch
import argparse
import numpy as np # Keep for potential type checks

# 注意：你的 Config 类在 utils/configurator.py 中
from utils.configurator import Config 
from utils.dataset_new import MyDataset # The class we are testing
from model.GDNSM import MyGDNSM

# main_1进行dataset适配

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
    dataset_info = test_dataset.get_dataset_info()
    model = MyGDNSM(config, dataset_info).to(config['device'])

    #test 外部接口
    print("\n[Step 2/4] Testing external interfaces...")
    try:
        trn_d, tst_d = test_dataset.get_pytorch_datasets()
        adj = test_dataset.get_adj_matrix()
        img_feat, txt_feat = test_dataset.get_all_features()
        
        print(f"  - Got trn_dataset of type: {type(trn_d)}")
        print(f"  - Got adj_matrix of shape: {adj.shape}")
        print(f"  - Got img_feat of shape: {img_feat.shape}")
        print("✅ SUCCESS: All getter methods work as expected.")
    except Exception as e:
        print(f"❌ FAILED: Error calling getter methods: {e}")