'''
Description: 
Author: Yu Sha
Date: 2022-05-09 17:41:12
LastEditors: Yu Sha
LastEditTime: 2023-10-25 15:49:22
'''

import torch
import numpy as np
import pickle
import numpy as np

# Generate Statistical Correlation Matrix ==== A matrix
def Generate_StatisticalCorrelationMatrix(num_classes, tau, eta, adj_file):
    adj_dict = pickle.load(open(adj_file, 'rb'))
    _adj = adj_dict['adj']
    return _adj

def update_adj_matrix(adj, row_indices, col_indices, value):
    for row in row_indices:
        for col in col_indices:
            adj[row][col] = value

# Generate Hierarchical Edge Knowledge Correlation Matrix ==== A matrix
def Generate_HierarchicalEdgeKnowledgeCorrelationMatrix(num_classes, tau, eta, adj_file):
    adj_dict = pickle.load(open(adj_file, 'rb'))
    _adj = adj_dict['adj']
    # ['chokedflow(0)', 'constant(1)', 'incipient(2)', 'non(3)', 'cavitation(4)']
    # choked/constant/incipient---->non
    row_to_zero = [0, 1, 2]
    col_to_zero = [3]
    update_adj_matrix(_adj, row_to_zero, col_to_zero, 0)
    # non---->choked/constant/incipient
    row_to_zero = [3]
    col_to_zero = [0, 1, 2]
    update_adj_matrix(_adj, row_to_zero, col_to_zero, 0)
    row_to_zero = [0, 1, 2]
    col_to_zero = [4]
    update_adj_matrix(_adj, row_to_zero, col_to_zero, 1)

    with open('./utils/cavitation_hierarchical_edge_knowledge_adj.pkl', 'wb') as file:
        pickle.dump(adj_dict, file, pickle.HIGHEST_PROTOCOL)
    return _adj

def Generate_BinaryCorrelationMatrix(num_classes, tau, eta, adj_file):
    adj_dict = pickle.load(open(adj_file, 'rb'))
    _adj = adj_dict['adj']
    # ['chokedflow(0)', 'constant(1)', 'incipient(2)', 'non(3)', 'cavitation(4)']
    # choked/constant/incipient---->non
    row_to_zero = [0, 1, 2]
    col_to_zero = [3]
    update_adj_matrix(_adj, row_to_zero, col_to_zero, 0)
    # non---->choked/constant/incipient
    row_to_zero = [3]
    col_to_zero = [0, 1, 2]
    update_adj_matrix(_adj, row_to_zero, col_to_zero, 0)
    row_to_zero = [0, 1, 2]
    col_to_zero = [4]
    update_adj_matrix(_adj, row_to_zero, col_to_zero, 1)
    
    _adj[_adj < tau] = 0
    _adj[_adj >= tau] = 1
    with open('./utils/cavitation_binary_adj.pkl', 'wb') as file:
        pickle.dump(adj_dict, file, pickle.HIGHEST_PROTOCOL)
    return _adj

def Generate_ReweithtCorrelationMatrix(num_classes, tau, eta, adj_file):
    adj_dict = pickle.load(open(adj_file, 'rb'))
    _adj = adj_dict['adj']
    # ['chokedflow(0)', 'constant(1)', 'incipient(2)', 'non(3)', 'cavitation(4)']
    # choked/constant/incipient---->non
    row_to_zero = [0, 1, 2]
    col_to_zero = [3]
    update_adj_matrix(_adj, row_to_zero, col_to_zero, 0)
    # non---->choked/constant/incipient
    row_to_zero = [3]
    col_to_zero = [0, 1, 2]
    update_adj_matrix(_adj, row_to_zero, col_to_zero, 0)
    row_to_zero = [0, 1, 2]
    col_to_zero = [4]
    update_adj_matrix(_adj, row_to_zero, col_to_zero, 1)

    _adj[_adj < tau] = 0
    _adj[_adj >= tau] = 1

    _adj = _adj * eta / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + (1 - eta)*np.identity(num_classes, np.int)
    with open('./utils/cavitation_reweight_adj.pkl', 'wb') as file:
        pickle.dump(adj_dict, file, pickle.HIGHEST_PROTOCOL)
    return _adj

# Generate_adj ==== A_hat matrix(normalized version of correlation matrix A)
def  Generate_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

