import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from opts import parse_opts
from engine import *
from gcn_layers import *
import timm

class ViT_ADD_GCN(nn.Module):
    def __init__(self, model_name, num_classes, in_channel=300, tau=0, eta=0, adj_file=None, pretrained=False):
        super(ViT_ADD_GCN, self).__init__()
        
        # Load the Vision Transformer (ViT) model
        self.features = timm.create_model(model_name, pretrained=pretrained)
        self.num_classes = num_classes
        
        # Adaptive pooling layer
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # Fully connected layer to transform the ViT output to match GCN input dimension
        self.fc = nn.Linear(1000, 512)

        # Define the GCN layers
        self.gcn1 = GraphConvolutionNetwork(in_channel, 512)
        self.gcn2 = GraphConvolutionNetwork(512, 512)
        
        # Activation functions
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # Get correlation matrix
        _adj = Generate_StatisticalCorrelationMatrix(num_classes, tau, eta, adj_file)
        self.CorrelationMatrix = Parameter(torch.from_numpy(_adj).float())

    def forward(self, feature, inp):
        # Pass the input through the ViT model
        feature = self.features(feature)
        
        # Handle different shapes of ViT output
        if len(feature.shape) == 3:  # (batch_size, num_patches, embedding_dim)
            feature = feature.mean(dim=1)  # Take the mean across the sequence length dimension
        elif len(feature.shape) == 2:  # (batch_size, embed_dim)
            pass  # feature already has the correct shape
        else:
            raise ValueError(f"Unexpected feature shape: {feature.shape}")

        # Transform the ViT output to match GCN input dimension
        feature = self.fc(feature)
        
        # GCN input: word embedding and correlation matrix
        inp = inp[0]
        adj = Generate_adj(self.CorrelationMatrix).detach()
        
        # Pass through GCN layers
        x = self.gcn1(inp, adj)
        x = self.leakyrelu(x)
        x = self.gcn2(x, adj)

        # Transpose and multiply with feature vector
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        x = self.sigmoid(x)
        return x

    def get_config_optim(self, lr, lrp):
        # Return the configuration for the optimizer
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.fc.parameters(), 'lr': lr},
            {'params': self.gcn1.parameters(), 'lr': lr},
            {'params': self.gcn2.parameters(), 'lr': lr},
        ]

# Function to create a ViT model with added GCN
def vit_tiny_patch16_224_add_gcn(num_classes, tau, eta, pretrained=False, adj_file=None, in_channel=300):
    return ViT_ADD_GCN("vit_tiny_patch16_224", num_classes, tau=tau, eta=eta, adj_file=adj_file, in_channel=in_channel, pretrained=pretrained)

def vit_small_patch16_224_add_gcn(num_classes, tau, eta, pretrained=False, adj_file=None, in_channel=300):
    return ViT_ADD_GCN("vit_small_patch16_224", num_classes, tau=tau, eta=eta, adj_file=adj_file, in_channel=in_channel, pretrained=pretrained)

def vit_small_patch32_224_add_gcn(num_classes, tau, eta, pretrained=False, adj_file=None, in_channel=300):
    return ViT_ADD_GCN("vit_small_patch32_224", num_classes, tau=tau, eta=eta, adj_file=adj_file, in_channel=in_channel, pretrained=pretrained)

def vit_base_patch8_224_add_gcn(num_classes, tau, eta, pretrained=False, adj_file=None, in_channel=300):
    return ViT_ADD_GCN("vit_base_patch8_224", num_classes, tau=tau, eta=eta, adj_file=adj_file, in_channel=in_channel, pretrained=pretrained)

def vit_base_patch16_224_add_gcn(num_classes, tau, eta, pretrained=False, adj_file=None, in_channel=300):
    return ViT_ADD_GCN("vit_base_patch16_224", num_classes, tau=tau, eta=eta, adj_file=adj_file, in_channel=in_channel, pretrained=pretrained)
