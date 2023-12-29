import time
import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch_sparse import spmm

import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score,\
                            f1_score,accuracy_score,precision_score,\
                            recall_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from tdc.multi_pred import DTI
from loguru import logger

from Utils import csv_record,check_dir,save_model,class_metrics

class SparseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Sparse Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """
    def __init__(self, in_channels, out_channels, iterations, dropout_rate, device):
        super(SparseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).to(self.device)
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels)).to(self.device)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        feature_count, _ = torch.max(features["indices"], dim=1)
        feature_count = feature_count + 1
        base_features = spmm(features["indices"], features["values"], feature_count[0],
                             feature_count[1], self.weight_matrix)

        base_features = base_features + self.bias

        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)

        base_features = torch.nn.functional.relu(base_features)
        for _ in range(self.iterations-1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        return base_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'

class DenseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Dense Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """
    def __init__(self, in_channels, out_channels, iterations, dropout_rate, device):
        super(DenseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).to(self.device)
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels)).to(self.device)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        base_features = torch.mm(features, self.weight_matrix)
        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)
        for _ in range(self.iterations-1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        base_features = base_features + self.bias
        return base_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'


class MixHopNetwork(torch.nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """

    def __init__(self,
                 feature_number, 
                 class_number=1,
                 layers_1=[32, 32, 32, 32], 
                 layers_2=[32, 32, 32, 32], 
                 hidden1=64,
                 hidden2=32,
                 dropout=0.1,
                 device=torch.device('cpu')):
        super(MixHopNetwork, self).__init__()
        self.layers_1 = layers_1
        self.layers_2 = layers_2
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.feature_number = feature_number
        self.class_number = class_number
        self.dropout = dropout
        self.device = device
        self.calculate_layer_sizes()
        self.setup_layer_structure()

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.layers_1)
        self.abstract_feature_number_2 = sum(self.layers_2)
        self.order_1 = len(self.layers_1)
        self.order_2 = len(self.layers_2)

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        self.upper_layers = [
            SparseNGCNLayer(self.feature_number, self.layers_1[i - 1], i, self.dropout, self.device) for
            i
            in range(1, self.order_1 + 1)]
        self.upper_layers = nn.ModuleList(self.upper_layers)

        self.bottom_layers = [
            DenseNGCNLayer(self.abstract_feature_number_1, self.layers_2[i - 1], i, self.dropout,
                           self.device) for i in
            range(1, self.order_2 + 1)]
        self.bottom_layers = nn.ModuleList(self.bottom_layers)

        self.bilinear = nn.Bilinear(self.abstract_feature_number_2, self.abstract_feature_number_2, self.hidden1)
        self.decoder = nn.Sequential(nn.Linear(self.hidden1, self.hidden2),
                                     nn.ELU(),
                                     nn.Linear(self.hidden2, 1)
                                     )

    def embed(self, normalized_adjacency_matrix, features):
        """
                Forward pass.
                :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
                :param features: Feature matrix.
                :return feat: higher order features
                """
        abstract_features_1 = torch.cat(
            [self.upper_layers[i](normalized_adjacency_matrix, features) for i in range(self.order_1)], dim=1)
        abstract_features_1 = F.dropout(abstract_features_1, self.dropout, training=self.training)

        abstract_features_2 = torch.cat(
            [self.bottom_layers[i](normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)],
            dim=1)
        feat = F.dropout(abstract_features_2, self.dropout, training=self.training)
        return feat

    def forward(self, normalized_adjacency_matrix, features, idx):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
                latent_features: latent representations of nodes
        """
        latent_features = self.embed(normalized_adjacency_matrix, features)

        feat_p1 = latent_features[idx[0]]
        feat_p2 = latent_features[idx[1]]
        feat = F.elu(self.bilinear(feat_p1, feat_p2))
        feat = F.dropout(feat, self.dropout, training=self.training)
        predictions = self.decoder(feat)
        return predictions, latent_features


class Data_DTI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[str(self.df.iloc[index].Drug_ID)]
        idx2 = self.idx_map[self.df.iloc[index].Target_ID]
        y = self.labels[index]
        return y, (idx1, idx2)

def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + 2 * I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(A, device):
    """
    Creating a propagator matrix.
    :param graph: NetworkX graph.
    :return propagator: Dictionary of matrix indices and values.
    """
    I = sp.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sp.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    propagator["indices"] = torch.LongTensor(ind.T).to(device)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data).to(device)
    return propagator

def features_to_sparse(features, device):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param path: Path to the JSON file.
    :return out_features: Dict with index and value tensor.
    """
    index_1, index_2 = features.nonzero()
    values = [1.0]*len(index_1)
    node_count = features.shape[0]
    feature_count = features.shape[1]
    features = sp.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T).to(device)
    out_features["values"] = torch.FloatTensor(features.data).to(device)
    out_features["dimensions"] = features.shape
    return out_features

def calc_score(model, data_loader,batch_size, propagation_matrix, features):
    model.eval()
    batch_total = len(data_loader)
    y_pred = np.empty([batch_total,batch_size])
    y_label = np.empty([batch_total,batch_size])
    for i in tqdm(range(batch_total), 'metrics'):
        label, pairs = next(iter(data_loader))
        output, latent_feature = model(propagation_matrix, features, pairs)
        label_ids = label.numpy()
        y_label[i] = label_ids
        y_pred[i] = output.flatten().detach().numpy()
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

def dti_data_preprocess(df, oversampling=True):
    df = df.dropna() # drop NaN rows
    df['Drug_ID'] = df['Drug_ID'].astype(str)
    df['Label'] = 1
    df['Label'][df.Y <= 30.0] = 0  # 30.0
    # print(df[df.Label == 0])
    # print(df[df.Label == 1])
    neg_samples = df[df.Label == 0]
    pos_samples =  df[df.Label == 1]
    neg_label_num = neg_samples.shape[0]
    pos_label_num = pos_samples.shape[0]
    # print(neg_samples)
    # print(pos_samples)
    logger.info(f'neg samples(0): {neg_label_num}, pos samples(1): {pos_label_num}, {neg_label_num * 100 //(neg_label_num + pos_label_num)}%')
    if oversampling:
        logger.info('oversampling')
        for _ in range(pos_label_num//neg_label_num):
            df = df.append(neg_samples,ignore_index=True)
        neg_samples = df[df.Label == 0]
        pos_samples =  df[df.Label == 1]
        neg_label_num = neg_samples.shape[0]
        pos_label_num = pos_samples.shape[0]
        logger.info(f'neg samples(0): {neg_label_num}, pos samples(1): {pos_label_num}, {neg_label_num * 100 //(neg_label_num + pos_label_num)}%')
    return df

def train(name, device=torch.device('cpu')):
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
    
    
    data_dti = DTI(name = name)
    split = data_dti.get_split(method = 'random', seed = 42, frac = [1.0, 0, 0])
    
    df = dti_data_preprocess(split['train'])
    logger.info(f"{name}: \n {df}")
    idx = df['Drug_ID'].tolist()+df['Target_ID'].tolist()
    idx = list(set(idx))
    idx = np.array(idx)
    idx_total = len(idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges_unordered = df[['Drug_ID', 'Target_ID']].values
    # print(f'edges_unordered:{len(edges_unordered)}')
    
    features = np.eye(idx_total)  # Drug_ID + Target_ID
    features = features_to_sparse(features, device)
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    # print(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)
    # print(adj)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device)
    
    # print(adj)
    
    propagation_matrix = adj
    features = features
    idx_map = idx_map
    
    split = data_dti.get_split(method = 'random', seed = 42, frac = [0.7, 0.1, 0.2])
    df_train =  dti_data_preprocess(split['train'])
    df_val = dti_data_preprocess(split['valid'])
    df_test = dti_data_preprocess(split['test'])
    
    train_params = {'batch_size': batch_size,
                        'shuffle': True,
                        # 'num_workers': 6,
                        'drop_last': True}

    test_params = {'batch_size': batch_size,
                    'shuffle': False,
                    # 'num_workers': 6,
                    'drop_last': True
                    }

    training_set = Data_DTI(idx_map, df_train.Label.values, df_train)
    train_loader = data.DataLoader(training_set, **train_params)

    validation_set = Data_DTI(idx_map, df_val.Label.values, df_val)
    val_loader = data.DataLoader(validation_set, **test_params)

    test_set = Data_DTI(idx_map, df_test.Label.values, df_test)
    test_loader = data.DataLoader(test_set, **test_params)
    
    feature_number = features["dimensions"][1]
    
    model = MixHopNetwork(feature_number)
    
    no_improvement = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    max_auc = 0
    # loss_history = []

    t_total = time.time()
    logger.info('Start Training...')
    for epoch in range(epochs):
        model.train()
        t = time.time()
        batch_total = len(train_loader)
        y_pred_train = np.empty([batch_total,batch_size])
        y_label_train = np.empty([batch_total,batch_size])
        epoch_loss = 0
        for i in tqdm(range(batch_total), f"train epoch{epoch + 1}"):
            label, pairs = next(iter(train_loader))
            model.train()
            optimizer.zero_grad()
            label = label.to(device)
            # logger.info(f'pairs:{pairs}')
            # logger.info(f'propagation_matrix:{propagation_matrix}')
            # logger.info(f'features:{features}')
            prediction, _ = model(propagation_matrix, features, pairs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction.squeeze(), label.float())
            # logger.info(f"prediction.squeeze():{prediction.squeeze()}")
            # logger.info(f'label.float():{label.float()}')
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            csv_record(csv_path+"hoagcn_loss.csv",{'epoch':epoch, 'batch':i, 'loss':loss.item(), 'avg_loss':epoch_loss/(i+1)})

            y_label_train[i] =  label.flatten().numpy()
            y_pred_train[i] =  prediction.detach().flatten().numpy()
        y_label_train = y_label_train.flatten()
        y_pred_train = y_pred_train.flatten()
        logger.info(f"y_label_train:{y_label_train}")
        logger.info(f"y_pred_train:{y_pred_train}")
        logger.info('Epoch: ' + str(epoch + 1) + '/' + str(epochs) + ' Iteration: ' + str(i + 1) + '/' +
                        str(len(train_loader)) + ' Training loss: ' + str(loss.cpu().detach().numpy()))
        roc_train = roc_auc_score(y_label_train, y_pred_train)

        # validation after each epoch
        result = calc_score(model, val_loader,batch_size, propagation_matrix, features)
        result['epoch'] = epoch
        result['epoch_loss'] = epoch_loss/batch_total
        csv_record(csv_path+"hoagcn_val_metrics.csv",result)
        logger.info(f'Train: {result}')
        roc_val,prc_val, f1_val = result['auroc'],result['auprc'],result['f1']
        if roc_val > max_auc:
            max_auc = roc_val
            no_improvement = 0
        else:
            no_improvement = no_improvement + 1
            if no_improvement == early_stopping:
                break

        logger.info('epoch: {:04d}, '.format(epoch + 1)+
                'loss_train: {:.4f}, '.format(loss.item())+
                'auroc_train: {:.4f}, '.format(roc_train)+
                'auroc_val: {:.4f}, '.format(roc_val)+
                'auprc_val: {:.4f}, '.format(prc_val)+
                'f1_val: {:.4f}, '.format(f1_val)+
                'time: {:.4f}s'.format(time.time() - t))

    logger.info("Optimization Finished!")
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # sava model
    save_model(model, model_path+f"hoagcn_{name}_epoch{epochs}.pt")
    
    # Testing
    result = calc_score(model, test_loader, batch_size, propagation_matrix, features)
    csv_record(csv_path+"hoagcn_test_metrics.csv",result)
    print(f'Test: {result}')
    logger.remove(log_fd)



if __name__ == '__main__':
    # train('BindingDB_Kd')
    # train('BindingDB_IC50')
    # train('BindingDB_Ki')
    train('DAVIS')
    # train('KIBA')