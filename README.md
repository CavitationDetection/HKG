# Hierarchical Knowledge Guided Fault Intensity Diagnosis of Complex Industrial Systems 

![framework](https://github.com/CavitationDetection/HKG/blob/main/images/framework.png)

## Requirements

- Python 3.8.11
- torch 1.9.1
- torchvision 0.10.1

Note: our model is trained on a SLURM-managed server node with 2011G RAM, 128-core CPUs and eight NVIDIA A100.

## Code execution

- train.py is the entry point to the code.
- main.py is the main function of our model.
- models/xxx.py is the network structure of our method (e.g. resnet_add_gcn.py, mobilenet_v2_add_gcn.py, vit_add_gcn.py and so on).
- opts.py is all the necessary parameters for our method (e.g. comprehensive output factor, learning rate and data loading path and so on).
- engine.py contains the construction of the different correlation matrices (e.g. SCM, HKCM, Binary HEKCM and Re-weighted HEKCM).
- gcn_layers.py is the network structure of GCN.
- train/test_data_loader.py represents the loading of training and test datasets.
- generate_adj_file.py indicates the generation of the adjacency matrix.
- generate_word_embedding.py is the generation of word embeddings for the target classes (e.g. GloVe, GoogleNews, FastText and so on).
- Execute train.py


## Test dataset
Download datasets from [here](https://drive.google.com/drive/folders/1eejPrqM2hWPxSfb0gUhu-F4FD0rhO7sp?usp=sharing) and place test signals in the subdirectories of ./Data/Test/



## Updates
[24.06.2024] Our code are all released.

[17.05.2024] Our manuscript has been accepted for the 30th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (ACM SIGKDD2024).

[08.02.2024] Our manuscript was submitted to the 30TH ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (ACM SIGKDD2024).

[18.02.2024] For the time being, some codes are being made available and the full codes will be released when the manuscript is accepted.

For any queries about codes, please feel free to contact YuSha et al. through yusha20211001@gmail.com

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{10.1145/3637528.3671610,
author = {Sha, Yu and Gou, Shuiping and Liu, Bo and Faber, Johannes and Liu, Ningtao and Schramm, Stefan and Stoecker, Horst and Steckenreiter, Thomas and Vnucec, Domagoj and Wetzstein, Nadine and Widl, Andreas and Zhou, Kai},
title = {Hierarchical Knowledge Guided Fault Intensity Diagnosis of Complex Industrial Systems},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3671610},
doi = {10.1145/3637528.3671610},
abstract = {Fault intensity diagnosis (FID) plays a pivotal role in monitoring and maintaining mechanical devices within complex industrial systems. As current FID methods are based on chain of thought without considering dependencies among target classes. To capture and explore dependencies, we propose a <u>h</u>ierarchical <u>k</u>nowledge <u>g</u>uided fault intensity diagnosis framework (HKG) inspired by the tree of thought, which is amenable to any representation learning methods. The HKG uses graph convolutional networks to map the hierarchical topological graph of class representations into a set of interdependent global hierarchical classifiers, where each node is denoted by word embeddings of a class. These global hierarchical classifiers are applied to learned deep features extracted by representation learning, allowing the entire model to be end-to-end learnable. In addition, we develop a re-weighted hierarchical knowledge correlation matrix (Re-HKCM) scheme by embedding inter-class hierarchical knowledge into a data-driven statistical correlation matrix (SCM) which effectively guides the information sharing of nodes in graphical convolutional neural networks and avoids over-smoothing issues. The Re-HKCM is derived from the SCM through a series of mathematical transformations. Extensive experiments are performed on four real-world datasets from different industrial domains (three cavitation datasets from SAMSON AG and one existing publicly) for FID, all showing superior results and outperform recent state-of-the-art FID methods.},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {5657–5668},
numpages = {12},
keywords = {acoustic signals, cavitation intensity diagnosis, hierarchical classification, hierarchical knowledge, representation learning and graph convolutional network},
location = {Barcelona, Spain},
series = {KDD '24}
}
```




