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
[17.05.2024] Our manuscript has been accepted for the 30th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (ACM SIGKDD2024).

[08.02.2024] Our manuscript was submitted to the 30TH ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (ACM SIGKDD2024).

[18.02.2024] For the time being, some codes are being made available and the full codes will be released when the manuscript is accepted.

## Citation
If you find our work useful in your research, please consider citing:



