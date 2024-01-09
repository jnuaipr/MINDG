import os
import time
import datetime
import random

import torch
from torch import nn, threshold
from torch.nn import functional as F
from torch.utils import data
import torch.optim.lr_scheduler as lr_scheduler

import scipy.sparse as sp
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score,\
                            f1_score,accuracy_score,auc,roc_curve
                            
import pandas as pd
from tqdm import tqdm
from loguru import logger
import pandas as pd
from tdc.multi_pred import DTI

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
# from DeepPurpose.utils import *
# from DeepPurpose.dataset import *

from HOAGCN import MixHopNetwork
from HDN import get_model
from Utils import csv_record,check_dir,save_model,load_model,class_metrics,features_to_sparse,create_propagator_matrix,setup_seed

neg_label = 1
pos_label = 0

class MINDG(nn.Module):
    def __init__(self, hdn_model, hoagcn_model,propagation_matrix, features, alpha= 0.9) -> None:
        super().__init__()
        self.view1_model = hdn_model
        self.view2_model = hoagcn_model
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.propagation_matrix = propagation_matrix
        self.features = features
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, v_d, v_p, idx1, idx2):
        pred1 = self.view1_model(v_d, v_p) # sequence
        pred2,_ = self.view2_model(self.propagation_matrix,self.features, (idx1, idx2)) # graph
        output = self.alpha * pred1 + (1 - self.alpha)* pred2
        return output
 
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

def get_threshold(label, pred):
    df_pred = pd.DataFrame(pred, columns=['pred'])
    df_pred = df_pred.sort_values(by=['pred'])
    df_label = pd.DataFrame(label, columns=['label'])
    neg_num = df_label[df_label.label == neg_label].shape[0]
    pos_num = df_label[df_label.label == pos_label].shape[0]
    threshold_idx = int(neg_num / (neg_num + pos_num) * df_pred.shape[0])
    threshold = df_pred.at[threshold_idx, 'pred']
    logger.info(f'threshold:{threshold}')
    return threshold
    
def calc_score(model, data_loader,batch_size):
    with torch.no_grad():
        model.eval()
        batch_total = len(data_loader)
        y_pred = np.empty([batch_total,batch_size])
        y_label = np.empty([batch_total,batch_size])
        for i in tqdm(range(batch_total), 'metrics'):
            v_d, v_p, y,idx_1,idx_2,label  = next(iter(data_loader))
            pred = model(v_d, v_p, idx_1,idx_2)
            label_ids = y.numpy()
            y_label[i] = label_ids
            y_pred[i] = pred.flatten().detach().numpy()
        y_pred = y_pred.flatten()
        y_label = y_label.flatten()
        threshold = 0.5 #get_threshold(y_label, y_pred)  #0.5
        y_pred_binary = np.empty(batch_total*batch_size)
        for i in range(len(y_pred_binary)):
            if y_pred[i] >  threshold:  # 
                y_pred_binary[i] = neg_label
            else:
                y_pred_binary[i] = pos_label
        auprc = average_precision_score(y_label, y_pred)
        auroc = roc_auc_score(y_label, y_pred)
        logger.info(f'y_label: {y_label}')
        logger.info(f'y_pred: {y_pred}')
        # f1 = f1_score(y_label, y_pred_binary)
        # recall  = recall_score(y_label, y_pred_binary)
        # sensitivity = recall
        # precision = precision_score(y_label, y_pred_binary)
        # accuracy = accuracy_score(y_label, y_pred_binary)
        result = class_metrics(y_label, y_pred_binary)
        result['auprc'] = auprc
        result['auroc'] = auroc
        # result['auc'] = auc_cindex
        return result

def sample_stat(df):
    neg_samples = df[df.Label == neg_label]
    pos_samples =  df[df.Label == pos_label]
    neg_label_num = neg_samples.shape[0]
    pos_label_num = pos_samples.shape[0]
    logger.info(f'neg/pos:{neg_label_num}/{pos_label_num}, neg:{neg_label_num * 100 //(neg_label_num + pos_label_num)}%, pos:{pos_label_num * 100 //(neg_label_num + pos_label_num)}%')
    return neg_label_num, pos_label_num

def find_unobserved_pair(df, drug_ids, target_ids):
    while(1):
        drug_id = random.sample(drug_ids, 1)[0]
        target_id = random.sample(target_ids, 1)[0]
        dfA = df[df.Drug_ID == drug_id]
        if target_id not in dfA["Target_ID"].values:
            break
    return drug_id, target_id
    

def get_unobserved_negative_samples(df):
    neg_samples = df[df.Label == neg_label]
    pos_samples =  df[df.Label == pos_label]
    neg_label_num = neg_samples.shape[0]
    pos_label_num = pos_samples.shape[0]
    delta = pos_label_num - neg_label_num
    drug_dict = {}
    target_dict = {}
    drug_ids = list(df['Drug_ID'].unique())
    target_ids = list(df['Target_ID'].unique())
    if len(drug_ids)*len(target_ids) < delta + pos_label_num + neg_label_num:
        iter_num = pos_label_num // neg_label_num
        for _ in range(iter_num):
            df = df.append(neg_samples, ignore_index=True)
    else:
        for id in tqdm(drug_ids, "drug dict"):
            drug = df[df.Drug_ID == id].Drug.values[0]
            drug_dict[id] = drug
        for id in tqdm(target_ids, "target dict"):
            target = df[df.Target_ID == id].Target.values[0]
            target_dict[id] = target
        for _ in tqdm(range(delta), "oversampling"):
            drug_id, target_id = find_unobserved_pair(df, drug_ids, target_ids)
            row = [drug_id, drug_dict[drug_id], target_id, target_dict[target_id], neg_label]
            df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
    sample_stat(df)
    return df

def df_data_preprocess(df, oversampling=False, undersampling=True):
    df = df.dropna() # drop NaN rows
    df['Drug_ID'] = df['Drug_ID'].astype(str)
    df = df.rename(columns={"Y": "Label"})
    neg_label_num, pos_label_num = sample_stat(df)
    if oversampling:
        logger.info('oversampling')
        pos_samples = df[df.Label == pos_label]
        for _ in range(1):
            df = df.append(pos_samples,ignore_index=True)
        # df = get_unobserved_negative_samples(df)
    if undersampling:
        logger.info('undersampling')
        neg_samples = df[df.Label == neg_label][:pos_label_num]
        pos_samples = df[df.Label == pos_label]
        df = pos_samples.append(neg_samples, ignore_index=True)
    sample_stat(df)
    return df

def df_data_split(df,frac=[0.7, 0.1, 0.2]):
    df = df.sample(frac=1, replace=True, random_state=1) # shuffle
    total = df.shape[0]
    train_idx = int(total*frac[0])
    valid_idx = int(total*(frac[0]+frac[1]))
    df_train = df.iloc[:train_idx]
    df_valid = df.iloc[train_idx:valid_idx]
    df_test = df.iloc[valid_idx:total-1]
    sample_stat(df_train)
    sample_stat(df_valid)
    sample_stat(df_test)
    return df_train, df_valid, df_test

def dti_df_process(df):
    df = df.dropna() # drop NaN rows
    seq_drug = df['Drug']
    seq_target = df['Target']
    seq_label = df['Label']
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

def run(name, phase="train",batch_size=32,epochs=5,learning_rate=5e-4,lr_step_size=10,early_stopping=10,device=torch.device('cpu'),seed_id=10):
    batch_size = 32
    epochs = 5# 5 10 20
    learning_rate = 5e-4
    lr_step_size = 10
    device = torch.device('cpu')
    
    setup_seed(seed_id)
    
    base_path = "/home/yang/sda/github/MINDG/"
    model_path = base_path+ f"output/model/"
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = base_path+f'output/{name}/'+now +'/'
    csv_path = root_path
    log_path = root_path
    check_dir(root_path)
    check_dir(csv_path)
    check_dir(model_path)
    check_dir(log_path)
    log_fd = logger.add(log_path+"/train.log")
    
    # generate ID map
    data_dti = DTI(name = name)
    if name in "DAVIS":
        data_dti.convert_to_log(form = 'binding')
        data_dti.binarize(threshold = 7, order = 'descending') # 7
    elif name == "BindingDB_Kd":
        data_dti.convert_to_log(form = 'binding')
        data_dti.binarize(threshold = 7.6, order = 'descending') #7.6
    elif name == "KIBA":
        data_dti.binarize(threshold = 12.1, order = 'descending')
    else:
        logger.error(f"dataset {name} is not supported")
        return
    df = data_dti.get_data()
    df = df_data_preprocess(df)
    logger.info(f"{name}: \n{df.head(5)}")
    idx = np.concatenate((df['Drug_ID'].unique(), df['Target_ID'].unique()))
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges_unordered = df[['Drug_ID', 'Target_ID']].values
    idx_total = len(idx)
    features = np.eye(idx_total)  # Drug_ID + Target_ID
    features = features_to_sparse(features, device)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device)
    propagation_matrix = adj
    
    # train/valid/test dataframe
    df_train, df_valid, df_test = df_data_split(df)
    df_train = dti_df_process(df_train)
    df_valid = dti_df_process(df_valid)
    df_test = dti_df_process(df_test)
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
    
    
    hdn_model = get_model().model
    feature_number = features["dimensions"][1]
    hoagcn_model = MixHopNetwork(feature_number)
    
    mindg_model = MINDG(hdn_model, hoagcn_model, propagation_matrix, features)
    if phase=='train' or not os.path.exists(model_path+f"mindg_{name}_epoch{epochs}.pt"):
        optimizer = torch.optim.Adam(mindg_model.parameters(), lr = learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1)
        logger.info('Start Training...')
        t_total = time.time()
        for epoch in range(epochs):
            t = time.time()
            epoch_loss = 0
            batch_total = len(train_loader)
            y_pred_train = np.empty([batch_total,batch_size])
            y_label_train = np.empty([batch_total,batch_size])
            for i in tqdm(range(batch_total), f"train epoch{epoch + 1}"):
                v_d, v_p, y,idx_1,idx_2,label = next(iter(train_loader))
                optimizer.zero_grad()
                pred = mindg_model( v_d, v_p, idx_1,idx_2)
                pred = pred.flatten()
                label = y.float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                csv_record(csv_path+"mindg_loss.csv",{'epoch':epoch+1, 'batch':i, 'loss':loss.item(), 'avg_loss':epoch_loss/(i+1)})
                y_label_train[i] =  label.flatten().numpy()
                y_pred_train[i] =  pred.detach().flatten().numpy()
            scheduler.step()
            y_label_train = y_label_train.flatten()
            y_pred_train = y_pred_train.flatten()
            logger.info(f'y_label_train:{y_label_train}')
            logger.info(f'y_pred_train:{y_pred_train}')
            logger.info('Epoch: ' + str(epoch + 1) + '/' + str(epochs) + ' Iteration: ' + str(i + 1) + '/' +
                            str(len(train_loader)) + ' Training loss: ' + str(loss.cpu().detach().numpy()))
            roc_train = roc_auc_score(y_label_train, y_pred_train)
            # validation after each epoch
            result = calc_score(mindg_model, valid_loader,batch_size)
            result['epoch'] = epoch +1
            result['epoch_loss'] = epoch_loss/batch_total
            result['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
            csv_record(csv_path+"mindg_val_metrics.csv",result)
            logger.info(f'Train: {result}')
            roc_val,prc_val, f1_val = result['auroc'],result['auprc'],result['f1']

            logger.info('epoch: {:04d}, '.format(epoch + 1)+
                    'auroc_train: {:.4f}, '.format(roc_train)+
                    'auroc_val: {:.4f}, '.format(roc_val)+
                    'auprc_val: {:.4f}, '.format(prc_val)+
                    'f1_val: {:.4f}, '.format(f1_val)+
                    'time: {:.4f}s'.format(time.time() - t))

        logger.info("Optimization Finished!")
        logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # sava model
        save_model(mindg_model, model_path+f"mindg_{name}_epoch{epochs}.pt")
    
    # Testing
    load_model(mindg_model, model_path+f"mindg_{name}_epoch{epochs}.pt")
    result = calc_score(mindg_model, test_loader, batch_size)
    csv_record(csv_path+"mindg_test_metrics.csv",result)
    print(f'Test: {result}')
    logger.remove(log_fd)


if __name__ == '__main__':
    run('DAVIS')
    # run('BindingDB_Kd')
    # run('KIBA')
    # run('BindingDB_IC50')
    # run('BindingDB_Ki')
    