import pandas as pd
from tdc.multi_pred import DTI
import os
import random
from tqdm import tqdm

def get_unobserved_negative_samples(df):
    neg_samples = df[df.Y == 1]
    pos_samples =  df[df.Y == 0]
    neg_label_num = neg_samples.shape[0]
    pos_label_num = pos_samples.shape[0]
    delta = pos_label_num - neg_label_num
    drug_dict = {}
    target_dict = {}
    drug_ids = list(df['Drug_ID'].unique())
    target_ids = list(df['Target_ID'].unique())
    for id in tqdm(drug_ids, "drug dict"):
        drug = df[df.Drug_ID == id].Drug.values[0]
        drug_dict[id] = drug
        return
    for id in tqdm(target_ids, "target dict"):
        target = df[df.Target_ID == id].Drug.values[0]
        target_dict[id] = target
    for _ in tqdm(range(delta), "oversampling"):
        drug_id, target_id = find_unobserved_pair(df, drug_ids, target_ids)
        df.loc[len(df.index)] = [drug_id, drug_dict[drug_id], target_id, target_dict[target_id], 0]
    return df

data_dti = DTI(name = "BindingDB_Kd")
data_dti.binarize(threshold = 30, order = 'descending')
df = data_dti.get_data()
get_unobserved_negative_samples(df)