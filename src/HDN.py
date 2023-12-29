import os
import time
import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score,\
                            f1_score,accuracy_score
from tqdm import tqdm
from loguru import logger
import pandas as pd
from tdc.multi_pred import DTI

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

from Utils import csv_record,check_dir,save_model,class_metrics

class DTI_Dataset(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, df):
        'Initialization'
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[str(self.df.iloc[index].Graph_Drug)]
        idx2 = self.idx_map[self.df.iloc[index].Graph_Target]
        label = self.df.iloc[index].Graph_Label
        drug_encoding = self.df.iloc[index].drug_encoding
        target_encoding = self.df.iloc[index].target_encoding
        v_d = drug_encoding
        v_p = utils.protein_2_embed(target_encoding)
        y = self.df.iloc[index].Seq_Label
        return v_d, v_p , y, idx1, idx2,label


def get_model():
    drug_encoding, target_encoding = 'MPNN', 'CNN'
    config = utils.generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 5, 
                         LR = 0.001, 
                         batch_size = 128,
                         hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3, 
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12]
                        )
    model = models.model_initialize(**config)
    return model

def calc_score(model, data_loader,batch_size):
    model.eval()
    batch_total = len(data_loader)
    y_pred = np.empty([batch_total,batch_size])
    y_label = np.empty([batch_total,batch_size])
    for i in tqdm(range(batch_total), 'metrics'):
        v_d, v_p, y,idx_1,idx_2,label  = next(iter(data_loader))
        pred = model(v_d, v_p)
        label_ids = y.numpy()
        y_label[i] = label_ids
        y_pred[i] = pred.flatten().detach().numpy()
    y_pred = y_pred.flatten()
    y_label = y_label.flatten()
    # y_max = y_pred.max()
    # y_min = y_pred.min()
    threshold = 0.5
    y_pred_binary = np.empty(batch_total*batch_size)
    for i in range(len(y_pred_binary)):
        if y_pred[i] >  threshold:
            y_pred_binary[i] = 1
        else:
            y_pred_binary[i] = 0
    auprc = average_precision_score(y_label, y_pred)
    auroc = roc_auc_score(y_label, y_pred)
    # f1 = f1_score(y_label, y_pred_binary)
    # recall  = recall_score(y_label, y_pred_binary)
    # sensitivity = recall
    # precision = precision_score(y_label, y_pred_binary)
    # accuracy = accuracy_score(y_label, y_pred_binary)
    result = class_metrics(y_label, y_pred_binary)
    result['auprc'] = auprc
    result['auroc'] = auroc
    return result

def id_df_process(df):
    df = df.dropna() # drop NaN rows
    df['Drug_ID'] = df['Drug_ID'].astype(str)
    df['Label'] = 1
    df['Label'][df.Y <= 30.0] = 0  # 30.0
    return df

def dti_df_process(df):
    df = df.dropna() # drop NaN rows
    seq_drug = df['Drug']
    seq_target = df['Target']
    seq_label = df['Label']  #df['Y']
    graph_drug = df['Drug_ID']
    graph_target = df['Target_ID']
    graph_label = df['Label']
    df = pd.DataFrame(zip(seq_drug, seq_target, seq_label, 
                          graph_drug, graph_target,graph_label))
    df.rename(columns={0:'Seq_Drug',
                        1: 'Seq_Target',
                        2: 'Seq_Label',
                        3:'Graph_Drug',
                        4:'Graph_Target',
                        5:'Graph_Label'}, 
                            inplace=True)
    drug_encoding, target_encoding = 'MPNN', 'CNN'
    df = utils.encode_drug(df, drug_encoding, column_name='Seq_Drug')
    df = utils.encode_protein(df, target_encoding, column_name='Seq_Target')
    return df

def train(name):
    batch_size = 32
    epochs = 20
    learning_rate = 5e-4
    early_stopping = 10
    
    base_path = "/home/yang/sda/github/MINDG/"
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = base_path+f'output/{name}/'+now +'/'
    csv_path = root_path
    log_path = root_path
    model_path = base_path+ f"output/model/"
    check_dir(root_path)
    check_dir(csv_path)
    check_dir(model_path)
    check_dir(log_path)
    log_fd = logger.add(log_path+"/train.log")
    
    # generate ID map
    data_dti = DTI(name = name)
    split = data_dti.get_split(method = 'random', seed = 42, frac = [1.0, 0, 0])
    df = id_df_process(split['train'])
    logger.info(f"{name}: \n{df.head(5)}")
    idx = np.concatenate((df['Drug_ID'].unique(), df['Target_ID'].unique()))
    idx_map = {j: i for i, j in enumerate(idx)}
    
    # train/valid/test dataframe
    split = data_dti.get_split(method = 'random', seed = 42, frac = [0.7, 0.1, 0.2])
    df_train = dti_df_process(id_df_process(split['train']))
    df_valid = dti_df_process(id_df_process(split['valid']))
    df_test = dti_df_process(id_df_process(split['test']))
    logger.info(f'train: {df_train.shape}')
    logger.info(f'valid: {df_valid.shape}')
    logger.info(f'test: {df_test.shape}')
    logger.info(f'df_train: \n {df_train.head(5)}')
    
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    # 'num_workers': 6,
                    'drop_last': True,
                    'collate_fn': utils.mpnn_collate_func
                    }

    test_params = {'batch_size': batch_size,
                    'shuffle': False,
                    # 'num_workers': 6,
                    'drop_last': True,
                    'collate_fn': utils.mpnn_collate_func
                    }

    train_dataset = DTI_Dataset(idx_map, df_train)
    train_loader = data.DataLoader(train_dataset, **train_params)

    valid_dataset = DTI_Dataset(idx_map, df_valid)
    valid_loader = data.DataLoader(valid_dataset, **test_params)

    test_dataset = DTI_Dataset(idx_map, df_test)
    test_loader = data.DataLoader(test_dataset, **test_params)
    

    # v_d, v_p, y,idx_1,idx_2,label = next(iter(train_dataset))
    # logger.info(f'v_d:{v_d}')
    # logger.info(f'v_p:{v_p}')
    # logger.info(f'y:{y}')
    # logger.info(f'idx_1:{idx_1}')
    # logger.info(f'idx_2:{idx_2}')
    # logger.info(f'label:{label}')

    
    drug_encoding, target_encoding = 'MPNN', 'CNN'
    config = utils.generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 5, 
                         LR = 0.001, 
                         batch_size = 128,
                         hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3, 
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12]
                        )
    hdn_model = models.model_initialize(**config).model
    logger.info(f"{hdn_model}")
    
    optimizer = torch.optim.Adam(hdn_model.parameters(), lr = learning_rate)
    # model.train(train_data, valid_data, test_data)
    # model.save_model(os.getcwd()+f'/model/deeppurpose_{name}')
    # model = models.model_pretrained(path_dir=os.getcwd()+f'/model/deeppurpose_{name}') # load local model
    # model = models.model_pretrained(model = 'MPNN_CNN_BindingDB') # networks download pretrained models
    
    logger.info('Start Training...')
    t_total = time.time()
    for epoch in range(epochs):
        hdn_model.train() # train stage
        t = time.time()
        epoch_loss = 0
        batch_total = len(train_loader)
        y_pred_train = np.empty([batch_total,batch_size])
        y_label_train = np.empty([batch_total,batch_size])
        for i in tqdm(range(batch_total), f"train epoch{epoch + 1}"):
            v_d, v_p, y,idx_1,idx_2,label = next(iter(train_loader))
            optimizer.zero_grad()
            drug, target =  v_d, v_p
            pred = hdn_model(drug, target)
            pred = pred.flatten()
            label = y.float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            csv_record(csv_path+"hdn_loss.csv",{'epoch':epoch, 'batch':i, 'loss':loss.item(), 'avg_loss':epoch_loss/(i+1)})
            y_label_train[i] =  label.flatten().numpy()
            y_pred_train[i] =  pred.detach().flatten().numpy()
        y_label_train = y_label_train.flatten()
        y_pred_train = y_pred_train.flatten()
        logger.info('Epoch: ' + str(epoch + 1) + '/' + str(epochs) + ' Iteration: ' + str(i + 1) + '/' +
                        str(len(train_loader)) + ' Training loss: ' + str(loss.cpu().detach().numpy()))
        roc_train = roc_auc_score(y_label_train, y_pred_train)
        # validation after each epoch
        result = calc_score(hdn_model, valid_loader,batch_size)
        result['epoch'] = epoch
        result['epoch_loss'] = epoch_loss/batch_total
        csv_record(csv_path+"hdn_val_metrics.csv",result)
        logger.info(f'Train: {result}')
        roc_val,prc_val, f1_val = result['auroc'],result['auprc'],result['f1']

        logger.info('epoch: {:04d}'.format(epoch + 1),
                'auroc_train: {:.4f}'.format(roc_train),
                'auroc_val: {:.4f}'.format(roc_val),
                'auprc_val: {:.4f}'.format(prc_val),
                'f1_val: {:.4f}'.format(f1_val),
                'time: {:.4f}s'.format(time.time() - t))

    logger.info("Optimization Finished!")
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # sava model
    save_model(hdn_model, model_path+f"hdn_{name}_epoch{epochs}.pt")
    
    # Testing
    result = calc_score(hdn_model, test_loader, batch_size)
    csv_record(csv_path+"hdn_test_metrics.csv",result)
    print(f'Test: {result}')
    logger.remove(log_fd)


if __name__ == '__main__':
    train('BindingDB_Kd')
    # train('BindingDB_IC50')
    # train('BindingDB_Ki')
    # train('DAVIS')
    # train('KIBA')
    