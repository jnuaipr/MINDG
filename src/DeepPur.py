from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
from tdc.multi_pred import DTI
import os

def get_deeppur_model():
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

def train(name):
    data = DTI(name = name)
    # data = DTI(name = 'BindingDB_Kd')
    # data = DTI(name = 'BindingDB_IC50')
    # data = DTI(name = 'BindingDB_Ki')
    # data = DTI(name = 'DAVIS')
    # data = DTI(name = 'KIBA')
    split = data.get_split(method = 'random', seed = 42, frac = [0.7, 0.1, 0.2])
    train_data = split['train']
    valid_data = split['valid']
    test_data = split['test']
    train_data.combine_first(valid_data)
    train_data.combine_first(test_data)
    df =train_data
    print(df.info())
    X_drugs = df['Drug']
    X_targets = df['Target']
    y = df['Y']
    drug_encoding, target_encoding = 'MPNN', 'CNN'
    train_data, valid_data, test_data = utils.data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)
    print(f'train: {train_data.shape}')
    print(f'valid: {valid_data.shape}')
    print(f'test: {test_data.shape}')

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
    # model.train(train_data, valid_data, test_data)
    model.save_model(os.getcwd()+f'/model/deeppurpose_{name}')
    model = models.model_pretrained(path_dir=os.getcwd()+f'/model/deeppurpose_{name}') # load local model
    model = models.model_pretrained(model = 'MPNN_CNN_BindingDB') # networks download pretrained models

if __name__ == '__main__':
    train('BindingDB_Kd')
    