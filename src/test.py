import torch

import numpy as np
import scipy.sparse as sp

device = torch.device('cpu')

features = np.eye(7343)
index_1, index_2 = features.nonzero()
print(index_1)
print(index_2)
values = [1.0]*len(index_1)
# print(values)
node_count = features.shape[0]
print(node_count)
feature_count = features.shape[1]
print(feature_count)
features = sp.coo_matrix((values, (index_1, index_2)),
                                shape=(node_count, feature_count),
                                dtype=np.float32)
print(features)
out_features = dict()
ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
out_features["indices"] = torch.LongTensor(ind.T).to(device)
out_features["values"] = torch.FloatTensor(features.data).to(device)
out_features["dimensions"] = features.shape

print(out_features)