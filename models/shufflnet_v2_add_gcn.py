import math
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Parameter
from opts import parse_opts
from utils import *
from engine import *
from gcn_layers import *

class ShuffleNet_v2_ADD_GCN(nn.Module):
    # in_channel:dim of word vector
    # tau: threshold
    # eta: scaling factor
    # adj_file: the above generate adj file
    def __init__(self, model, num_classes, in_channel = 300, tau = 0, eta = 0, adj_file = None):
        super(ShuffleNet_v2_ADD_GCN, self).__init__()

        self.features = model.features
        self.num_classes = num_classes
        self.pooling = nn.AdaptiveAvgPool2d(output_size = (1, 1))

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

    # feature is CNN input, inp is word embedding
    def forward(self, feature, inp):
        # CNN input--->feature
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0),-1)
        
        # GCN input--->word embedding and correlation matrix
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

# representation learning is ShuffleNet_v2
def shufflenet_v2_x0_5_add_gcn(num_classes, tau, eta, pretrained = False, adj_file = None, in_channel = 300):
    model = models.shufflenet_v2_x0_5(pretrained = pretrained)
    return ShuffleNet_v2_ADD_GCN(model, num_classes, tau = tau, eta = eta, adj_file = adj_file, in_channel = in_channel)

def shufflenet_v2_x1_0_add_gcn(num_classes, tau, eta, pretrained = False, adj_file = None, in_channel = 300):
    model = models.shufflenet_v2_x1_0(pretrained = pretrained)
    return ShuffleNet_v2_ADD_GCN(model, num_classes, tau = tau, eta = eta, adj_file = adj_file, in_channel = in_channel)

def shufflenet_v2_x1_5_add_gcn(num_classes, tau, eta, pretrained = False, adj_file = None, in_channel = 300):
    model = models.shufflenet_v2_x1_5(pretrained = pretrained)
    return ShuffleNet_v2_ADD_GCN(model, num_classes, tau = tau, eta = eta, adj_file = adj_file, in_channel = in_channel)

def shufflenet_v2_x2_0_add_gcn(num_classes, tau, eta, pretrained = False, adj_file = None, in_channel = 300):
    model = models.shufflenet_v2_x2_0(pretrained = pretrained)
    return ShuffleNet_v2_ADD_GCN(model, num_classes, tau = tau, eta = eta, adj_file = adj_file, in_channel = in_channel)


