# main_2.py
# A refactored entry point for running a single experiment with the new dataset module.

import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

# --- 1. Import Core Modules ---
from utils.configurator import Config
from utils.logger import init_logger
from logging import getLogger
from utils.utils import init_seed, get_model, get_trainer # Assuming get_model/get_trainer exist

# --- 2. Import OUR NEW Dataset Module ---
from utils.dataset_new import MyDataset, TrnData, TstData 
from model.GDNSM import MyGDNSM
# We need TrnData/TstData for the negative sampling call

# --- 3. Import the Trainer we are building ---
from common.trainer import Trainer 

def run_single_experiment(config):
    """
    Runs a single, complete training and evaluation experiment.
    This function is a simplified, linear version of the original quick_start.
    """
    # --- A. Setup: Logger and Seed ---
    init_logger(config)
    logger = getLogger()
    init_seed(config['seed'])
    logger.info(config)

    # --- B. [MODIFIED] Data Loading ---
    logger.info("--> 1. Initializing MyDataset...")
    dataset = MyDataset(config)
    trn_dataset_obj, tst_dataset_obj = dataset.get_pytorch_datasets()
    logger.info("--> MyDataset Initialized.")

    logger.info("--> 2. Creating DataLoaders...")
    train_loader = DataLoader(trn_dataset_obj, batch_size=config['train_batch_size'], shuffle=True)
    test_loader = DataLoader(tst_dataset_obj, batch_size=config['eval_batch_size'], shuffle=False)
    logger.info("--> DataLoaders Created.")

    # --- C. [MODIFIED] Model Initialization ---
    logger.info("--> 3. Initializing Model...")
    # adj_matrix = dataset.get_adj_matrix()
    # features = dataset.get_all_features()
    dataset_info = dataset.get_dataset_info()
    model = MyGDNSM(config, dataset_info).to(config['device'])
    logger.info(model)
    
    # --- D. Optimizer and Trainer Initialization ---
    logger.info("--> 4. Initializing Optimizer and Trainer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # We use our custom Trainer here, assuming it takes these args.
    # Note: We need to adapt this if our custom Trainer's __init__ is different.
    trainer = Trainer(config, model, train_loader, test_loader, optimizer)
    logger.info("--> All components initialized. Starting training...")

    # --- E. Run Training ---
    # We need to add the negative sampling call before each epoch.
    # This logic will be moved into the Trainer's fit loop.
    trainer.fit() # Assuming the fit method handles the loop and neg sampling.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GDNSM', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    args = parser.parse_args()

    config = Config(model=args.model, dataset=args.dataset)
    
    run_single_experiment(config)