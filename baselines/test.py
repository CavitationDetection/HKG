import os
import argparse
import torch
import torch.nn as nn
from models.resnet import ResNet_18, ResNet_34
from testdataset import TestImgDataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description ='ResNet classification')
parser.add_argument('--cpu', action ='store_true', help = 'Choose device to use cpu or gpu (cuda:0)')
parser.add_argument('--test_root', action = 'store', type = str, default = '/home/deepthinkers/samson/yusha_workspace/Cavitation2017_STFT/Test/STFT/466944_10', help='Root path of image')
parser.add_argument('--test_label_path', action = 'store', type = str, default = '/home/deepthinkers/samson/yusha_workspace/Cavitation2017_STFT/Label/number_label/test_split_466944_10.csv', help='label file path')   
parser.add_argument('--batch_size', action = 'store', type = int, default = 16, help = 'number of data in a batch')
# feature extrector model
class FeatureExtractor(nn.Module):
    def __init__(self, orginial_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(orginial_model.children())[:-2])
    
    def forward(self, x):
        return self.features(x)


def test (opts):
    # select device, that is cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # test datset setting
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    # load  testing dataset
    dataset_test = TestImgDataset(opts, transform = test_transform)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = opts.batch_size, shuffle = False, num_workers = 1)

    # set model
    model = ResNet_34()
    model_path = "./model_result/modelResNet_34_20240119_91_0.8742857142857143.pth"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # get model name from the model path
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # create tearure extractor
    feature_extractor_model = FeatureExtractor(model)

    featurtes = []
    labels = []

    with torch.no_grad():
        feature_extractor_model.to(device)
        feature_extractor_model.eval()

        for test_img, test_label in data_loader_test:

            test_img = test_img.to(device)
            test_label = test_label.to(device)

            feature = feature_extractor_model(test_img)
            featurtes.append(feature.cpu())
            labels.append(test_label.cpu())

    featurtes = torch.cat(featurtes, dim = 0)
    labels = torch.cat(labels, dim = 0)
    torch.save(featurtes, "./output_results/feautres_{model_name}.pth")
    torch.save(labels, "./output_results/labels_{model_name}.pth")

if __name__ == "__main__":
    opts = parser.parse_args()
    test(opts)



