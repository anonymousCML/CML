import numpy as np
import scipy.sparse as sp
from scipy.sparse import *
import torch

from Params import args


def get_use(behaviors_data):

    behavior_mats = {}
        
    behaviors_data = (behaviors_data != 0) * 1

    behavior_mats['A'] = matrix_to_tensor(normalize_adj(behaviors_data))
    behavior_mats['AT'] = matrix_to_tensor(normalize_adj(behaviors_data.T))
    behavior_mats['A_ori'] = None

    return behavior_mats


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    rowsum_diag = sp.diags(np.power(rowsum+1e-8, -0.5).flatten())

    colsum = np.array(adj.sum(0))
    colsum_diag = sp.diags(np.power(colsum+1e-8, -0.5).flatten())

    
    return adj


def matrix_to_tensor(cur_matrix):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()  
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
    values = torch.from_numpy(cur_matrix.data)  
    shape = torch.Size(cur_matrix.shape)

    return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  