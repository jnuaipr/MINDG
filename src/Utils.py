import csv
import os

import torch

from loguru import logger

def csv_record(path, data):
    all_header = ['epoch','batch','loss','avg_loss','epoch_loss','auprc', 'auroc', 'sensitivity', 'specificity',
              'recall','precision','accuracy','f1']  
    row = []
    header = []
    for name in all_header:
        if name in data.keys():
            row.append(data[name])
            header.append(name)
            
    if os.path.exists(path):
        with open(path, 'a',newline='') as f:
            write = csv.writer(f)
            write.writerow(row)
    else:
        with open(path, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
            
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"create dir: {path}")
    else:
        logger.info(f"dir exists, {path}")
        
def save_model(model, path):
    torch.save(model.state_dict(),path)
    logger.info(f"save {path} model parameters done")

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"load {path} model parameters done")