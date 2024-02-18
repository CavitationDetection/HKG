# Hierarchical Knowledge Guided Fault Intensity Diagnosis of Complex Industrial Systems 

![framework](https://github.com/CavitationDetection/HKG/blob/main/images/framework.png)

## Requirements

- Python 3.8.11
- torch 1.9.1
- torchvision 0.10.1

Note: our model is trained on NVIDIA GPU (A100).

## Code execution

- train.py is the entry point to the code.
- main.py is the main function of our model.
- networks/xxx.py is the network structure of our method (e.g. resnet_add_gcn.py, mobilenet_v2_add_gcn.py, vit_add_gcn.py and so on).
- opts.py is all the necessary parameters for our method (e.g. comprehensive output factor, learning rate and data loading path and so on).
- engine.py contains the construction of the different correlation matrices (e.g. SCM, HKCM, Binary HEKCM and Re-weighted HEKCM).
- gcn_layers.py is the network structure of GCN.
- train/test_data_loader.py represents the loading of training and test datasets.
- generate_adj_file.py indicates the generation of the adjacency matrix.
- generate_word_embedding.py is the generation of word embeddings for the target classes (e.g. GloVe, GoogleNews, FastText and so on).
- Execute train.py

Note that, for the current version. test.py is nor required as the code calls the test function every iteration from within to visualize the performance difference between the baseline and the GRLNet. However, we also provide a separate test.py file for visualising the test set. For that, the instructions can be found below.

- Download datasets from [here](https://drive.google.com/drive/folders/1eejPrqM2hWPxSfb0gUhu-F4FD0rhO7sp?usp=sharing) and place test signals in the subdirectories of ./Data/Test/
- run test.py


## Updates
[08.02.2024] Our manuscript was submitted to the 30TH ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (ACM SIGKDD2024).

[18.02.2024] For the time being, some codes are being made available and the full codes will be released when the manuscript is accepted.




