import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from opts import parse_opts
from utils import *
from engine import *
from gcn_layers import *
import timm

class SwinTransformer_ADD_GCN(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, tau=0, eta=0, adj_file=None):
        super(SwinTransformer_ADD_GCN, self).__init__()

        self.features = model
        self.blocks = nn.Sequential(*list(self.features.children())[:-1])
        self.final_block = list(self.features.children())[-1]

        self.num_classes = num_classes
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # Fully connected layer to transform the ViT output to match GCN input dimension
        self.fc = nn.Linear(1000, 512)
        self.gcn1 = GraphConvolutionNetwork(in_channel, 512)
        self.gcn2 = GraphConvolutionNetwork(512, 512)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        _adj = Generate_StatisticalCorrelationMatrix(num_classes, tau, eta, adj_file)
        self.CorrelationMatrix = Parameter(torch.from_numpy(_adj).float())

    def forward(self, feature, inp):
        feature = self.blocks(feature)
        feature = self.final_block(feature)
        # Handle different shapes of ViT output
        if len(feature.shape) == 3:  # (batch_size, num_patches, embedding_dim)
            feature = feature.mean(dim=1)  # Take the mean across the sequence length dimension
        elif len(feature.shape) == 2:  # (batch_size, embed_dim)
            pass  # feature already has the correct shape
        else:
            raise ValueError(f"Unexpected feature shape: {feature.shape}")

        # Transform the ViT output to match GCN input dimension
        feature = self.fc(feature)

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

def swin_tiny_patch4_window7_224_add_gcn(num_classes, tau, eta, pretrained=False, adj_file=None, in_channel=300):
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained)
    return SwinTransformer_ADD_GCN(model, num_classes, tau=tau, eta=eta, adj_file=adj_file, in_channel=in_channel)

def swin_small_patch4_window7_224_add_gcn(num_classes, tau, eta, pretrained=False, adj_file=None, in_channel=300):
    model = timm.create_model("swin_small_patch4_window7_224", pretrained=False)
    return SwinTransformer_ADD_GCN(model, num_classes, tau=tau, eta=eta, adj_file=adj_file, in_channel=in_channel)

def swin_base_patch4_window7_224_add_gcn(num_classes, tau, eta, pretrained=False, adj_file=None, in_channel=300):
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False)
    return SwinTransformer_ADD_GCN(model, num_classes, tau=tau, eta=eta, adj_file=adj_file, in_channel=in_channel)

def swin_base_patch4_window12_384_add_gcn(num_classes, tau, eta, pretrained=False, adj_file=None, in_channel=300):
    model = timm.create_model("swin_base_patch4_window12_384", pretrained=False)
    return SwinTransformer_ADD_GCN(model, num_classes, tau=tau, eta=eta, adj_file=adj_file, in_channel=in_channel)
