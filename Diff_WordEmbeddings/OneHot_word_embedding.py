import torch
import numpy as np
import pickle

# Mock vocabulary and word indices
vocab = ['non', 'cavitation', 'choked', 'flow', 'constant', 'incipient']
word_indices = {word: i for i, word in enumerate(vocab)}

# Function to convert a word to its One-Hot representation
def one_hot_encode(word, vocab_size):
    one_hot = torch.zeros(vocab_size)
    one_hot[word_indices[word]] = 1
    return one_hot

# Function to calculate cosine similarity for One-Hot encoded vectors
def cosine(x, y):
    cosine = torch.dot(x, y) / (torch.norm(x) * torch.norm(y))
    return cosine

if __name__ == '__main__':
    num_classes = 5
    total = np.array([])

    # Represent words using One-Hot encoding
    non_cavitation = one_hot_encode('non', len(vocab)) + one_hot_encode('cavitation', len(vocab))
    total = np.append(total, non_cavitation.numpy())

    cavitation = one_hot_encode('cavitation', len(vocab))
    total = np.append(total, cavitation.numpy())

    choked_flow_cavitation = one_hot_encode('choked', len(vocab)) + \
                             one_hot_encode('flow', len(vocab)) + \
                             one_hot_encode('cavitation', len(vocab))
    total = np.append(total, choked_flow_cavitation.numpy())

    constant_cavitation = one_hot_encode('constant', len(vocab)) + one_hot_encode('cavitation', len(vocab))
    total = np.append(total, constant_cavitation.numpy())

    incipient_cavitation = one_hot_encode('incipient', len(vocab)) + one_hot_encode('cavitation', len(vocab))
    total = np.append(total, incipient_cavitation.numpy())

    # num_classes=5, because we have 4 classes
    total = total.reshape(num_classes, -1)

    # save One-Hot encoded vectors for the corresponding class
    with open('./utils/cavitation_onehot.pkl', 'wb') as f:
        pickle.dump(total, f, protocol=4)

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
