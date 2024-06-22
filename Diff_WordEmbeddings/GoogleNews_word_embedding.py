import torch
from gensim.models import KeyedVectors
import numpy as np
import pickle

# 计算余弦相似度
def cosine(x, y):
    cosine = torch.matmul(x, y.view((-1,))) / ((torch.sum(x * x) + 1e-9).sqrt() * torch.sum(y * y).sqrt())
    return cosine

if __name__ == '__main__':
    num_classes = 5
    total = np.array([])

    # 使用gensim加载Google News预训练的词向量
    google_news_model = KeyedVectors.load_word2vec_format('F:\WorkSpace_Cavitation\GCNNet_vfinal\.vector_cache\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin', binary=True)

    # non cavitation==health
    non = torch.tensor(google_news_model['non'])
    cavitation = torch.tensor(google_news_model['cavitation'])
    non_cavitation = non + cavitation
    total = np.append(total, non_cavitation.numpy())

    # cavitation
    total = np.append(total, cavitation.numpy())

    # choked flow cavitation
    choked = torch.tensor(google_news_model['choked'])
    flow = torch.tensor(google_news_model['flow'])
    choked_flow_cavitation = choked + flow + cavitation
    total = np.append(total, choked_flow_cavitation.numpy())

    # constant cavitation
    constant = torch.tensor(google_news_model['constant'])
    constant_cavitation = constant + cavitation
    total = np.append(total, constant_cavitation.numpy())

    # incipient cavitation
    incipient = torch.tensor(google_news_model['incipient'])
    incipient_cavitation = incipient + cavitation
    total = np.append(total, incipient_cavitation.numpy())

    # num_classes=5，因为我们有4个类别
    total = total.reshape(num_classes, -1)

    # 保存对应类别的词嵌入
    with open('./utils/cavitation_google_news_word2vec.pkl', 'wb') as f:
        pickle.dump(total, f, protocol=4)

    # 打印余弦相似度
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
