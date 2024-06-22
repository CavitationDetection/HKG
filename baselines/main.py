import argparse
import os
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from numpy import interp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from traindataset import TrainImgDataset
from testdataset import TestImgDataset
import torchvision.transforms as transforms
from datetime import datetime
from models.mobile_v2 import MobileNet_v2
from models.mobile_v3 import MobileNet_v3_S, MobileNet_v3_L
from models.shufflnet_v2 import ShuffleNet_v2_x0_5,ShuffleNet_v2_x1_0,ShuffleNet_v2_x1_5,ShuffleNet_v2_x2_0
from models.resnet import ResNet_18, ResNet_34, ResNet_50, ResNet_101, ResNet_152
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from models.squeezenet import SqueezeNet1_0, SqueezeNet1_1
from models.vit import vit_tiny_patch16_224, vit_small_patch16_224, vit_small_patch32_224, vit_base_patch32_224, vit_base_patch16_224
from models.swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224, swin_base_patch4_window12_384
import torchvision.utils as vutils

parser = argparse.ArgumentParser(description='classification')

parser.add_argument('--cuda:0', action='store_true', help='Choose device to use cpu cuda:0')
parser.add_argument('--arch', default = 'ResNet_18', type = str, 
                    help = 'Name of model to train')
parser.add_argument('--train_root', action='store', type=str,
                        default='/home/deepthinkers/samson/yusha_workspace/Cavitation2017_STFT/Train/STFT/466944_10', help='Root path of image')
parser.add_argument('--train_label_path', action='store', type=str, 
                        default='/home/deepthinkers/samson/yusha_workspace/Cavitation2017_STFT/Label/number_label/train_split_466944_10.csv', help='label file path')

parser.add_argument('--test_root', action='store', type=str,
                        default='/home/deepthinkers/samson/yusha_workspace/Cavitation2017_STFT/Test/STFT/466944_10', help='Root path of image')
parser.add_argument('--test_label_path', action='store', type=str, 
                        default='/home/deepthinkers/samson/yusha_workspace/Cavitation2017_STFT/Label/number_label/test_split_466944_10.csv', help='label file path')                       
parser.add_argument('--dropout', action='store', type=int, 
                        default=0.2, help='prob')
parser.add_argument('--batch_size', action='store', type=int, 
                        default = 16, help='number of data in a batch')
parser.add_argument('--lr', action='store', type=float, 
                        default = 0.1, help='initial learning rate')
parser.add_argument('--epochs', action='store', type=int, 
                        default = 150, help='train rounds over training set')
parser.add_argument('--num_classes', action='store', type=int, 
                        default = 4, help='total number of classes in dataset')
parser.add_argument('--expectedacc', action = 'store', type = float, default = 0.75,
                        help = 'expected accuracy of the test set for model saving')
parser.add_argument('--expectedacc1', action = 'store', type = float, default = 0.85,
                        help = 'expected accuracy of the test set for model saving')


def create_dirs(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def train(opts):
    time = datetime.now().date().strftime('%Y%m%d')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 1.0),
    transforms.RandomVerticalFlip(p = 1.0),
    transforms.RandomRotation(degrees = [180, 180]),
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

    test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

    # load dataset
    dataset_train = TrainImgDataset(opts, transform = train_transform)
    dataset_test = TestImgDataset(opts, transform = test_transform)
    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=opts.batch_size, shuffle=True, num_workers=1)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=opts.batch_size, shuffle=False, num_workers=1)

    # get the model
    model_dict = {
        'ResNet_18': ResNet_18,
        'ResNet_34': ResNet_34, 
        'ResNet_50': ResNet_50, 
        'ResNet_101': ResNet_101, 
        'ResNet_152': ResNet_152,
        'vgg11': vgg11,
        'vgg11_bn': vgg11_bn,
        'vgg13': vgg13,
        'vgg13_bn': vgg13_bn,
        'vgg16': vgg16, 
        'vgg16_bn': vgg16_bn, 
        'vgg19': vgg19, 
        'vgg19_bn': vgg19_bn,
        'MobileNet_v2': MobileNet_v2,
        'MobileNet_v3_S': MobileNet_v3_S, 
        'MobileNet_v3_L': MobileNet_v3_L,
        'ShuffleNet_v2_x0_5': ShuffleNet_v2_x0_5,
        'ShuffleNet_v2_x1_0': ShuffleNet_v2_x1_0,
        'ShuffleNet_v2_x1_5': ShuffleNet_v2_x1_5,
        'ShuffleNet_v2_x2_0': ShuffleNet_v2_x2_0,
        'DenseNet121': DenseNet121,
        'DenseNet161': DenseNet161, 
        'DenseNet169': DenseNet169, 
        'DenseNet201': DenseNet201,
        'SqueezeNet1_0': SqueezeNet1_0, 
        'SqueezeNet1_1': SqueezeNet1_1,
        'vit_tiny_patch16_224': vit_tiny_patch16_224,     
        'vit_small_patch16_224': vit_small_patch16_224, 
        'vit_small_patch32_224': vit_small_patch32_224, 
        'vit_base_patch32_224': vit_base_patch32_224, 
        'vit_base_patch16_224': vit_base_patch16_224,
        'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224, 
        'swin_small_patch4_window7_224': swin_small_patch4_window7_224, 
        'swin_base_patch4_window7_224': swin_base_patch4_window7_224, 
        'swin_base_patch4_window12_384': swin_base_patch4_window12_384
    }
    if opts.arch in model_dict:
        model_func = model_dict[opts.arch]
        model = model_func().to(device)
    else:
        raise NotImplementedError
    if torch.cuda.device_count() > 1:
        print("We use two gpus",torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adamax(model.parameters(),lr=opts.lr,betas=(0.9,0.999),eps=1e-8)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    loss_fct = CrossEntropyLoss()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(opts.epochs):
        train_batch_num = 0
        train_loss = 0.0
        model.train()
        counts=0
        for img, label in data_loader_train:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fct(pred, label.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_batch_num += 1
            train_loss += loss.item()
            predict = pred.argmax(dim=1)
            counts += predict.cpu().eq(label.cpu()).sum().item()
        avg_acc = counts * 1.0 / len(data_loader_train.dataset)
        train_loss_list.append(train_loss / len(data_loader_train.dataset))
        train_acc_list.append(avg_acc)
        # write csv file
        train_loss_dataframe = pd.DataFrame(data=train_loss_list)
        train_acc_dataframe = pd.DataFrame(data=train_acc_list)
        train_loss_dataframe.to_csv('./output_results/train_loss_{name}_{date}.csv'.format(name = opts.arch, date = time), index=False)
        train_acc_dataframe.to_csv('./output_results/train_accuracy_{name}_{date}.csv'.format(name = opts.arch, date = time), index=False)

        model.eval()
        test_y = []
        test_y_pred = []
        counts=0
        test_loss = 0
        test_batch_num = 0
        outs = []
        labels = []
        with torch.no_grad():
            for test_img, test_label in data_loader_test:
                test_img = test_img.to(device)
                test_label = test_label.to(device)
                t_pred = model(test_img)
                outs.append(t_pred.cpu())
                labels.append(test_label.cpu())
                # accuracy
                loss = loss_fct(t_pred, test_label.view(-1))
                test_loss += loss.item()
                test_batch_num += 1
                test_y += list(test_label.data.cpu().numpy().flatten())
                test_y_pred += list(t_pred.data.cpu().numpy().flatten())
                predict = t_pred.argmax(dim=1)
                counts += predict.cpu().eq(test_label.cpu()).sum().item()

        outs = torch.cat(outs,dim=0)
        labels = torch.cat(labels).reshape(-1)
        avg_acc = counts * 1.0 / len(data_loader_test.dataset)
        test_acc_list.append(avg_acc)
        test_loss_list.append(test_loss / len(data_loader_test.dataset))
        print('epoch: %d, train loss: %.4f, test loss: %.4f,test accuracy: %.4f' %
            (epoch, train_loss / train_batch_num, test_loss/ test_batch_num,avg_acc))
        # write csv file
        test_loss_dataframe = pd.DataFrame(data = test_loss_list)
        test_acc_dataframe = pd.DataFrame(data = test_acc_list)
        test_loss_dataframe.to_csv('./output_results/test_loss_{name}_{date}.csv'.format(name = opts.arch, date = time), index=False)
        test_acc_dataframe.to_csv('./output_results/test_accuracy_{name}_{date}.csv'.format(name = opts.arch, date = time), index=False)

        pre = outs.detach().numpy()
        classes = pre.shape[1]
        y_test = torch.nn.functional.one_hot(labels, classes).numpy()
        predictions = pre.argmax(axis = -1)
        truelabel = y_test.argmax(axis = -1)
        cm = confusion_matrix(y_true = truelabel, y_pred = predictions)
        # precision recall f1
        precision_micro = precision_score(truelabel,predictions,average='micro')
        recall_micro = recall_score(truelabel,predictions,average='micro')
        f1score_micro = f1_score(truelabel,predictions,average='micro')

        precision_macro = precision_score(truelabel,predictions,average='macro')
        recall_macro = recall_score(truelabel,predictions,average='macro')
        f1score_macro = f1_score(truelabel,predictions,average='macro')
        accuracyscore = accuracy_score(truelabel,predictions)
        classificationreport = classification_report(truelabel,predictions)
        print("Classification Report:\n{}".format(classificationreport))
        print("Confusion Matrix:\n{}".format(cm))
        print("Accuracy:{}".format(accuracyscore))
        print('Precision(micro):{}'.format(precision_micro))
        print('Precision(macro):{}'.format(precision_macro))

        print("Recall(micro):{}".format(recall_micro))
        print("Recall(macro):{}".format(recall_macro))

        print("F1_score(micro):{}".format(f1score_micro))
        print("F1_score(macro):{}".format(f1score_macro))

        if avg_acc >= opts.expectedacc and avg_acc <= opts.expectedacc1:
            torch.save(model.state_dict(), "./model_result/model{name}_{date}_{epoch}_{value}.pth".format(
                name = opts.arch, date = time, epoch = epoch, value = accuracyscore))

if __name__ == "__main__":
    opts = parser.parse_args() # Namespace object
    create_dirs('./results')
    create_dirs('./output_results')
    create_dirs('./figs')
    create_dirs('./model_result')
    train(opts)