'''
Description: Get our data's word embedding, in here we use Glove method. 
            Of course, you can use FastText, GoogleNews and other methods
            signal---non cavitation==health
                  ---cavitaion--incipient
                              --constant
                              --choked flow
Author: Yu Sha
Date: 2023-04-07 08:54:38
LastEditTime: 2023-10-25 11:37:25
LastEditors: Yu Sha
'''
import torch
import torchtext.vocab as vocab
import numpy as np
import pickle


# calculate cosine similarity
def cosine(x, y):
    cosine = torch.matmul(x, y.view((-1,))) / ((torch.sum(x * x) + 1e-9).sqrt() * torch.sum(y * y).sqrt())
    return cosine

if __name__ == '__main__':
    num_classes = 5
    total = np.array([])
    # select the set of word vectors you need
    '''
    ['glove.42B.300d',
    'glove.840B.300d',
    'glove.twitter.27B.25d',
    'glove.twitter.27B.50d',
    'glove.twitter.27B.100d',
    'glove.twitter.27B.200d',
    'glove.6B.50d',
    'glove.6B.100d',  
    'glove.6B.200d',
    'glove.6B.300d']
    '''
    glove = vocab.GloVe(name = '6B', dim = 300)

    # glove.stoi[]---get index subscript of corresponding word vector
    # glove.vectors[]---get word vector corresponding to the subscript of word vector

    # non cavitation==health
    health = glove.vectors[glove.stoi['health']]
    # cavitation = glove.vectors[glove.stoi['cavitation']]
    # non_cavitation = non + cavitation
    non_cavitation = health 
    total = np.append(total, non_cavitation.numpy())

    # cavitation
    cavitation = glove.vectors[glove.stoi['cavitation']]
    total = np.append(total, cavitation.numpy())

    # choked flow cavitation
    choked = glove.vectors[glove.stoi['choked']]
    flow = glove.vectors[glove.stoi['flow']]
    cavitation = glove.vectors[glove.stoi['cavitation']]
    choked_flow_cavitation = choked + flow + cavitation
    total = np.append(total, choked_flow_cavitation.numpy())

    # constant cavitation
    constant = glove.vectors[glove.stoi['constant']]
    cavitation = glove.vectors[glove.stoi['cavitation']]
    constant_cavitation = constant + cavitation
    total = np.append(total, constant_cavitation.numpy())

    # incipient cavitation
    incipient = glove.vectors[glove.stoi['incipient']]
    cavitation = glove.vectors[glove.stoi['cavitation']]
    incipient_cavitation = incipient + cavitation
    total = np.append(total, incipient_cavitation.numpy())

    # num_classes=5, because we have 4 classes
    total = total.reshape(num_classes, -1)

    # save word embedding for the corresponding class
    with open('./utils/cavitation_glove_word2vec.pkl', 'wb') as f:
        pickle.dump(total, f, protocol = 4)

    # print cosine similarity
    print("Non Cavitation---Non Cavitation", cosine(non_cavitation, non_cavitation))
    print("Non Cavitation---Cavitation", cosine(non_cavitation, cavitation))
    print("Cavitation---Incipient Cavitation", cosine(cavitation, incipient_cavitation))
    print("Cavitation---Constant Cavitation", cosine(cavitation, constant_cavitation))
    print("Cavitation---Choked Flow Cavitation", cosine(cavitation, choked_flow_cavitation))
    print("Non Cavitation---Incipient Cavitation", cosine(non_cavitation, incipient_cavitation))
    print("Non Cavitation---Constant Cavitation", cosine(non_cavitation, constant_cavitation))
    print("Non Cavitation---Choked Flow Cavitation", cosine(non_cavitation, choked_flow_cavitation))
    print("Incipient Cavitation---Constant Cavitation", cosine(incipient_cavitation, constant_cavitation))
    print("Incipient Cavitation---Choked Flow Cavitation", cosine(incipient_cavitation, choked_flow_cavitation))
    print("Constant Cavitation---Choked Flow Cavitation", cosine(constant_cavitation, choked_flow_cavitation))



