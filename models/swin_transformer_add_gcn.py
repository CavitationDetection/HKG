'''
Description: Representation learning (Swin-Transformer)+ Graph Convolutional Network (GCN)
Author: Yu Sha
Date: 2022-04-02 19:30:52
LastEditors: Yu Sha
LastEditTime: 2023-10-26 10:45:22
'''
import math
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Parameter
from timm.models import swin_small_patch4_window7_224, swin_tiny_patch4_window7_224, swin_base_patch4_window7_224
from opts import parse_opts
from utils import *
from engine import *
from gcn_layers import *

class SwinTransformer_ADD_GCN(nn.Module):
    # in_channel:dim of word vector
    # tau: threshold
    # eta: scaling factor
    # adj_file: the above generate adj file
    def __init__(self, model, num_classes, in_channel = 300, tau = 0, eta = 0, adj_file = None):
        super(SwinTransformer_ADD_GCN, self).__init__()

        self.features = model.features
        self.num_classes = num_classes
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.gcn1 = GraphConvolutionNetwork(in_channel, 512)
        self.gcn2 = GraphConvolutionNetwork(512, 512)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # get corrrelation matrix
        _adj = Generate_StatisticalCorrelationMatrix(num_classes, tau, eta, adj_file)
        # _adj = Generate_hierarchicalEdgeKnowledgeCorrelationMatrix(num_classes, tau, eta, adj_file)
        # _adj = Generate_BinaryCorrelationMatrix(num_classes, tau, eta, adj_file)
        # _adj = Generate_ReweithtCorrelationMatrix(num_classes, tau, eta, adj_file)
        self.CorrelationMatrix = Parameter(torch.from_numpy(_adj).float())

    # feature is ViT input, inp is word embedding
    def forward(self, feature, inp):
        # ViT input--->feature
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        # ViT input--->word embedding and correlation matrix
        inp = inp[0]
        adj = Generate_adj(self.CorrelationMatrix).detach()
        x = self.gcn1(inp, adj)
        x = self.leakyrelu(x)
        x = self.gcn2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        x = self.sigmoid(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gcn1.parameters(), 'lr': lr},
                {'params': self.gcn2.parameters(), 'lr': lr},
                ]
"""
vit_b_16/32: b-->base model(depth: 12 Transformer layer, Embedding size:768, Multi-head self-attention: 12, Feedforward size: 3072)
            16/32: patch size
vit_l_16/32: l-->large model(depth: 24 Transformer layer, Embedding size:1024, Multi-head self-attention: 16, Feedforward size: 4096)
            16/32: patch size
vit_h_14:   h-->huge model(depth: 32 Transformer layer, Embedding size:1280, Multi-head self-attention: 16, Feedforward size: 5120)
            14: patch size
"""
# representation learning is Swin-Transformer
def swin_tiny_patch4_window7_224_add_gcn(num_classes, tau, eta, pretrained = False, adj_file = None, in_channel = 300):
    model = swin_tiny_patch4_window7_224(pretrained = pretrained)
    return SwinTransformer_ADD_GCN(model, num_classes, tau = tau, eta = eta, adj_file = adj_file, in_channel = in_channel)

def swin_small_patch4_window7_224_add_gcn(num_classes, tau, eta, pretrained = False, adj_file = None, in_channel = 300):
    model = swin_small_patch4_window7_224(pretrained = pretrained)
    return SwinTransformer_ADD_GCN(model, num_classes, tau = tau, eta = eta, adj_file = adj_file, in_channel = in_channel)

def swin_base_patch4_window7_224_add_gcn(num_classes, tau, eta, pretrained = False, adj_file = None, in_channel = 300):
    model = swin_base_patch4_window7_224(pretrained = pretrained)
    return SwinTransformer_ADD_GCN(model, num_classes, tau = tau, eta = eta, adj_file = adj_file, in_channel = in_channel)