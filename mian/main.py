'''
Description: 
Author: Yu Sha
Date: 2023-04-07 08:54:38
LastEditTime: 2023-10-26 11:54:00
LastEditors: Yu Sha
'''
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support, accuracy_score,confusion_matrix
from datetime import datetime
from utils import *
from opts import *
from engine import *
from train_data_loader import TrainDataset
from test_data_loader import TestDataset
from models.densenet_add_gcn import *
from models.mobilenet_v2_add_gcn import *
from models.mobilenet_v3_add_gcn import *
from models.resnet_add_gcn import *
from models.shufflnet_v2_add_gcn import *
from models.squeezenet_add_gcn import *
from models.swin_transformer_add_gcn import *
from models.vgg_add_gcn import *
from models.vit_add_gcn import *
import warnings
warnings.filterwarnings("ignore")

# main function include train and test
def train(opts):
    time = datetime.datetime.now().date().strftime('%Y%m%d')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 1.0),
    transforms.RandomVerticalFlip(p = 1.0),
    transforms.RandomRotation(degrees = [180, 180]),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    # load dataset
    dataset_train = TrainDataset(opts, transform = train_transform, word_embedding_name = opts.word_embedding)
    dataset_test = TestDataset(opts, transform = test_transform, word_embedding_name = opts.word_embedding)

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, 
                                                    batch_size = opts.batch_size, 
                                                    shuffle = opts.train_batch_shuffle, 
                                                    num_workers = opts.n_threads,
                                                    drop_last = opts.train_drop_last)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                                    batch_size = opts.batch_size, 
                                                    shuffle = opts.test_batch_shuffle, 
                                                    num_workers = opts.n_threads,
                                                    drop_last = opts.test_drop_last)    

    # get model
    model_dict = {
        'resnet18_add_gcn': resnet18_add_gcn,
        'resnet34_add_gcn': resnet34_add_gcn,
        'resnet50_add_gcn': resnet50_add_gcn,
        'resnet101_add_gcn': resnet101_add_gcn,
        'vgg11_add_gcn': vgg11_add_gcn,
        'vgg11_bn_add_gcn': vgg11_bn_add_gcn,
        'vgg13_add_gcn': vgg13_add_gcn,
        'vgg13_bn_add_gcn': vgg13_bn_add_gcn,
        'vgg16_add_gcn': vgg16_add_gcn,
        'vgg16_bn_add_gcn': vgg16_bn_add_gcn,
        'vgg19_add_gcn': vgg19_add_gcn,
        'vgg19_bn_add_gcn': vgg19_bn_add_gcn,
        'mobilenet_v2_add_gcn': mobilenet_v2_add_gcn,
        'mobilenet_v3_small_add_gcn': mobilenet_v3_small_add_gcn,
        'mobilenet_v3_large_add_gcn': mobilenet_v3_large_add_gcn,
        'shufflenet_v2_x0_5_add_gcn': shufflenet_v2_x0_5_add_gcn,
        'shufflenet_v2_x1_0_add_gcn': shufflenet_v2_x1_0_add_gcn,
        'shufflenet_v2_x1_5_add_gcn': shufflenet_v2_x1_5_add_gcn,
        'shufflenet_v2_x2_0_add_gcn': shufflenet_v2_x2_0_add_gcn,
        'squeezenet1_0_add_gcn': squeezenet1_0_add_gcn,
        'squeezenet1_1_add_gcn': squeezenet1_1_add_gcn,
        'densenet121_add_gcn': densenet121_add_gcn,
        'densenet161_add_gcn': densenet161_add_gcn,
        'densenet169_add_gcn': densenet169_add_gcn,
        'densenet201_add_gcn': densenet201_add_gcn,
        'vit_tiny_patch16_224_add_gcn': vit_b_16_add_gcn,
        'vit_small_patch16_224_add_gcn': vit_b_32_add_gcn,
        'vit_small_patch32_224_add_gcn': vit_l_16_add_gcn,
        'vit_base_patch8_224_add_gcn': vit_l_32_add_gcn,
        'vit_base_patch16_224_add_gcn': vit_h_14_add_gcn,
        'swin_tiny_patch4_window7_224_add_gcn': swin_tiny_patch4_window7_224_add_gcn,
        'swin_small_patch4_window7_224_add_gcn': swin_small_patch4_window7_224_add_gcn,
        'swin_base_patch4_window7_224_add_gcn': swin_base_patch4_window7_224_add_gcn
    }
    if opts.arch in model_dict:
        model_func = model_dict[opts.arch]
        model = model_func(
            num_classes = opts.num_classes, 
            tau = opts.prob_threshold, 
            eta = opts.scaling_factor,
            adj_file = opts.adj_file_path
        ).to(device)
    else:
        raise NotImplementedError

    # for name, param in model.named_parameters():
    #     print(f"Layer: {name}, Data Type: {param.dtype}")
    # define optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr = opts.learning_rate)
    optimizer = torch.optim.SGD(model.get_config_optim(opts.learning_rate, opts.learning_rate_pretrained), 
                                lr = opts.learning_rate,
                                momentum = opts.momentum,
                                weight_decay = opts.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), 
    #                             lr = opts.learning_rate,
    #                             momentum = opts.momentum,
    #                             weight_decay = opts.weight_decay)
    print(" Start Training")

    # calculate acc
    loss_train_list = []
    acc_train_liist = []
    acc_train_modified_list = []
    loss_test_list = []
    acc_test_list = []
    acc_test_modified_list = []
    output_train_threshold_list = []
    output_train_original_list = []
    output_train_modified_original_list = []
    output_train_modified_threshold_list = []
    
    for num_epoch in range(opts.epoches):
        train_iter = 0
        loss_train = 0
        correct = 0
        correct_modified = 0
        model.train()

        for idx, (input_train, target_train) in enumerate(data_loader_train):

            cnn_input_train = input_train[0].to(device)
            gcn_input_train = input_train[2].to(device)
            target_train = target_train.to(device)

            optimizer.zero_grad()

            output_train = model(cnn_input_train, gcn_input_train)
            loss = F.multilabel_soft_margin_loss(output_train,target_train)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_iter += 1
            loss_train += loss.item()

            # multi-label results
            auc_thresholds = choose_threshold_based_on_auc(target_train, output_train)
            threshold = [auc_thresholds[i] for i in range(output_train.shape[1])]
            predicted_train = (output_train >= torch.tensor(threshold, device = output_train.device)).to(torch.int)
            correct += (predicted_train.cpu().eq(target_train.cpu()).all(dim=1)).sum().item()
            # save multi-label results
            output_train_threshold_list.append(predicted_train.cpu().detach().numpy())
            output_train_original_list.append(output_train.cpu().detach().numpy())


            # final hierarchical results
            output_train_modified = torch.cat((output_train[:, :1], output_train[:, 2:]), dim=1)
            target_train_modified = torch.cat((target_train[:, :1], target_train[:, 2:]), dim=1)

            auc_thresholds_modified = choose_threshold_based_on_auc(target_train_modified, output_train_modified)
            threshold_modified = [auc_thresholds_modified[i] for i in range(output_train_modified.shape[1])]
            predicted_train_modified = (output_train_modified >= torch.tensor(threshold_modified, device = output_train_modified.device)).to(torch.int)
            correct_modified += (predicted_train_modified.cpu().eq(target_train_modified.cpu()).all(dim=1)).sum().item()

            # save final hierarchical results
            output_train_modified_threshold_list.append(predicted_train_modified.cpu().detach().numpy())
            output_train_modified_original_list.append(output_train_modified.cpu().detach().numpy())
            
        accuracy_train = 1* correct / len(data_loader_train.dataset)
        accuracy_train_modified = 1* correct_modified / len(data_loader_train.dataset)
        loss_train_list.append(loss_train / len(data_loader_train.dataset))
        acc_train_liist.append(accuracy_train)
        acc_train_modified_list.append(accuracy_train_modified)

        # multi-label results
        output_train_threshold = np.concatenate(output_train_threshold_list)
        output_train_original = np.concatenate(output_train_original_list)

        # final hierarchical results
        output_train_modified_threshold = np.concatenate(output_train_modified_threshold_list)
        output_train_modified_original = np.concatenate(output_train_modified_original_list)

        # write csv file
        loss_train_dataframe = pd.DataFrame(data = loss_train_list)
        acc_train_dataframe = pd.DataFrame(data = acc_train_liist)
        output_train_threshold_dataframe = pd.DataFrame(data = output_train_threshold)
        output_train_original_dataframe = pd.DataFrame(data = output_train_original)
        output_train_modified_threshold_dataframe = pd.DataFrame(data = output_train_modified_threshold)
        output_train_modified_original_dataframe = pd.DataFrame(data = output_train_modified_original)

        loss_train_dataframe.to_csv('./output_results/train_loss_{name}_{date}.csv'.format(name = opts.arch, date = time))
        acc_train_dataframe.to_csv('./output_results/train_acc_{name}_{date}.csv'.format(name = opts.arch, date = time))
        output_train_threshold_dataframe.to_csv('./output_results/train__threshold_output_{name}_{date}.csv'.format(name = opts.arch, date = time, index = False))
        output_train_original_dataframe.to_csv('./output_results/train_original_output_{name}_{date}.csv'.format(name = opts.arch, date = time, index = False))
        output_train_modified_threshold_dataframe.to_csv('./output_results/train_modified_threshold_output_{name}_{date}.csv'.format(name = opts.arch, date = time, index = False))
        output_train_modified_original_dataframe.to_csv('./output_results/train_modified_original_output_{name}_{date}.csv'.format(name = opts.arch, date = time, index = False))

        model.eval()
        output_test_threshold_list = []
        output_test_original_list = []
        output_test_modified_threshold_list = []
        output_test_modified_original_list = []
        label_test = []
        label_test_modified = []
        result_test = []
        loss_test = 0
        correct_test = 0
        correct_test_modified = 0
        test_iter = 0
        with torch.no_grad():
            for idx, (input_test, target_test) in enumerate(data_loader_test):
                cnn_input_test = input_test[0].to(device)
                gcn_input_test = input_test[2].to(device)
                target_test = target_test.to(device)

                output_test = model(cnn_input_test, gcn_input_test)
                loss = F.multilabel_soft_margin_loss(output_test, target_test)
                loss_test += loss.item()

                # multi-label results
                auc_thresholds = choose_threshold_based_on_auc(target_test, output_test)
                threshold = [auc_thresholds[i] for i in range(output_test.shape[1])]
                pred_test = (output_test >= torch.tensor(threshold, device = output_test.device)).to(torch.int)

                # save multi-label results
                output_test_threshold_list.append(pred_test.cpu().detach().numpy())
                output_test_original_list.append(output_test.cpu().detach().numpy())
                result_test.append(pred_test.cpu().detach().numpy())
                label_test.append(target_test.cpu().detach().numpy())

                # final hierarchical results
                output_test_modified = torch.cat((output_test[:, :1], output_test[:, 2:]), dim=1)
                target_test_modified = torch.cat((target_test[:, :1], target_test[:, 2:]), dim=1)

                auc_thresholds_modified =choose_threshold_based_on_auc(target_test_modified, output_test_modified)
                threshold_modified = [auc_thresholds_modified[i] for i in range(output_test_modified.shape[1])]
                pred_test_modified = (output_test_modified >= torch.tensor(threshold_modified, device = output_test_modified.device)).to(torch.int)
                correct_test_modified += (pred_test_modified.cpu().eq(target_test_modified.cpu()).all(dim=1)).sum().item()

                # save hierarchical results
                output_test_modified_threshold_list.append(pred_test_modified.cpu().detach().numpy())
                output_test_modified_original_list.append(output_test_modified.cpu().detach().numpy())
                label_test_modified.append(target_test_modified.cpu().detach().numpy())

        loss_test_list.append(loss_test / len(data_loader_test.dataset))
        acc_test = correct_test / len(data_loader_test.dataset)
        acc_test_modified = correct_test_modified / len(data_loader_test.dataset)
        acc_test_list.append(acc_test)
        acc_test_modified_list.append(acc_test_modified)

        # multi-label results
        output_test_threshold = np.concatenate(output_test_threshold_list)
        output_test_original = np.concatenate(output_test_original_list)

        result_test = np.concatenate(result_test)
        label_test = np.concatenate(label_test)

        # final hierarchcial results
        output_test_modified_threshold = np.concatenate(output_test_modified_threshold_list)
        output_test_modified_original = np.concatenate(output_test_modified_original_list)
        label_test_modified = np.concatenate(label_test_modified)

        # write csv file
        loss_test_dataframe = pd.DataFrame(data = loss_test_list)
        acc_test_dataframe = pd.DataFrame(data = acc_test_list)
        output_test_threshold_dataframe = pd.DataFrame(data = output_test_threshold)
        output_test_original_dataframe = pd.DataFrame(data = output_test_original)
        output_test_modified_threshold_dataframe = pd.DataFrame(data = output_test_modified_threshold)
        output_test_modified_original_dataframe = pd.DataFrame(data = output_test_modified_original)

        loss_test_dataframe.to_csv('./output_results/test_loss_{name}_{date}.csv'.format(name = opts.arch, date = time))
        acc_test_dataframe.to_csv('./output_results/test_acc_{name}_{date}.csv'.format(name = opts.arch, date = time))
        output_test_threshold_dataframe.to_csv('./output_results/test_threshold_output_{name}_{date}.csv'.format(name = opts.arch, date = time, index = False))
        output_test_original_dataframe.to_csv('./output_results/test_original_output_{name}_{date}.csv'.format(name = opts.arch, date = time, index = False))
        output_test_modified_threshold_dataframe.to_csv('./output_results/test_modified_threshold_output_{name}_{date}.csv'.format(name = opts.arch, date = time, index = False))
        output_test_modified_original_dataframe.to_csv('./output_results/test_modified_original_output_{name}_{date}.csv'.format(name = opts.arch, date = time, index = False))
        
        ## metrics (multi-label)
        # confusion matrix
        confusion_matrices_multilabel = multilabel_confusion_matrix(label_test, result_test)

        # CP(per-class Precision), CR(per-class Recall), CF1（per-class F1-Score）
        per_class_metrics = precision_recall_fscore_support(label_test, result_test, average = None)
        CP = per_class_metrics[0]
        CR = per_class_metrics[1]
        CF1 = per_class_metrics[2]

        # ACP(average per-class Precision), ACR(average per-class Recall), ACF1（average per-class F1-Score）
        average_per_class_metrics = precision_recall_fscore_support(label_test, result_test, average = 'macro')
        ACP = average_per_class_metrics[0]
        ACR = average_per_class_metrics[1]
        ACF1 = average_per_class_metrics[2]

        # OP(average ovreall Precision）、OR（average ovreall Recall）、OF1（average ovreall F1-Score）
        average_ovreall_metrics = precision_recall_fscore_support(label_test, result_test, average = 'micro')
        AOP = average_ovreall_metrics[0]
        AOR = average_ovreall_metrics[1]
        AOF1 = average_ovreall_metrics[2]

        # Accuracy
        accuracy_test_multilabl = accuracy_score(label_test, result_test)

        # mAP (mean average precision)
        mAP = calculate_mAP(label_test, result_test)

        # ['CP', 'CR', 'CF1'] are writen in a '.csv' file
        write_metrics_to_csv([CP, CR, CF1], ['CP', 'CR', 'CF1'])

        ## metrics (final hierarchical)
        # confusion matrix
        label_test_modified = np.argmax(label_test_modified, axis = 1)
        output_test_modified_threshold = np.argmax(output_test_modified_threshold, axis = 1)
        confusion_matrices = confusion_matrix(label_test_modified, output_test_modified_threshold)
        accuracy_final = accuracy_score(label_test_modified, output_test_modified_threshold)
        precision_final, recall_final, f1_final, _ = precision_recall_fscore_support(label_test_modified, output_test_modified_threshold, average = 'weighted')

        print("Train--epoch:{0}|Train Loss:{1}|Test Loss:{2}|Train Accuracy:{3}|Test Accuracy:{4}|mAP:{5}|Accuracy(Final):{6}".format(
            num_epoch, loss_train / (train_iter if train_iter != 0 else 1), loss_test / (test_iter if test_iter != 0 else 1), accuracy_train, accuracy_test_multilabl, mAP, accuracy_final))
        
        print("Test--epoch:{0}, ACP:{1}, ACR:{2}, ACF1:{3}, AOP:{4}, AOR:{5}, AOF1:{6}, Accuracy:{7}, mAP:{8}, Accuracy(Final):{9}".format(
            num_epoch, ACP, ACR, ACF1, AOP, AOR, AOF1, accuracy_test_multilabl, mAP, accuracy_final))
        
        print("Multi-Label Confusion Matrix:\n{}".format(confusion_matrices_multilabel))
        print("Acc:{}".format(accuracy_test_multilabl))
        print("mAP:{}".format(mAP))
        print("ACP:{}".format(ACP))
        print("ACR:{}".format(ACR))
        print("AcF1:{}".format(ACF1))
        print("AOP:{}".format(AOP))
        print("AOR:{}".format(AOR))
        print("AOF1:{}".format(AOF1))
        print("-----------------------------------------------------------------------------------------------------------------------------------------------")
        print("Confusion Matrix(Final):\n{}".format(confusion_matrices))
        print("Acc(Final):{}".format(accuracy_final))
        print("F1-score:{}".format(f1_final))
        print("Recall(Final):{}".format(recall_final))
        print("Precision(Final):{}".format(precision_final))
        print("-----------------------------------------------------------------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------------------------------------------------------------")

        metrics = {
            "Epoch":num_epoch,
            "CP": CP,
            "CR": CR,
            "CF1": CF1,
            "ACP": ACP,
            "ACR": ACR,
            "ACF1": ACF1,
            "AOP": AOP,
            "AOR": AOR,
            "AOF1": AOF1,
            "Accuracy": accuracy_test_multilabl,
            "mAP": mAP,
            "Confusion Matrix(Final)": confusion_matrices,
            "Acc(Final)": accuracy_final,
            "F1-score": f1_final,
            "Recall(Final)": recall_final,
            "Precision(Final)": precision_final
        }
        log_metrics_to_file(metrics, './output_results/metrics.log')

        if accuracy_final >= opts.expectedacc:
            torch.save(model.state_dict(), "./model_result/model{name}_{date}_{epoch}_{value}.pth".format(
                name = opts.arch, data = time, epoch = num_epoch, value = accuracy_final))












