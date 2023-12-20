import torch
from torch.utils import data

import scipy.sparse as sp
import numpy as np
import pandas as pd

from HOGCN import MixHopNetwork
from DeepPur import get_deeppur_model
from tdc.multi_pred import DTI


if __name__ == '__main__':
    print('done')