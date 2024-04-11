Dear Reviewer,

Thank you very much for your thorough review and valuable feedbacks on our paper. We greatly appreciate your insights, and we will carefully consider each of your raised point and make improvements and clarifications in the revisions.

1.&nbsp;**Technical Soundness:** Missing baselines.

**Response:** Thank you for your assessment of the technical soundness of our paper. Our experimental design includes several baseline comparisons that can be used to evaluate the effectiveness of our proposed method. However, as you mentioned, there are still some baseline experiments missing.

For the three private real-world cavitation datasets provided by SAMSON AG (Cavitation-Short, Cavitation-Long and Cavitation-Noise), we employed a range of commonly used neural network architectures as baselines, including ResNet, VGG, DenseNet, MobileNet, ShuffleNet, ViT, and Swin Transformer. These baselines encompass Convolutional Neural Networks (CNN), Lightweight Neural Networks (LNN) and Transformer, providing a comprehensive assessment of the effectiveness of our proposed method. According to your suggestion and enhancing the completeness of our experiments, we have provided the results of different methods from cavitation intensity recognition or fault diagnosis (including non-hierarchical methods and hierarchical methods), as shown in Tables 1-3.

**Table 1: Results of different evaluation metrics on Cavitation-Short.**
|Methods|<center>Image size|<center>Acc</center>|<center>Pre</center>|<center>Rec</center>|<center>F1</center>|
|-|-|-|-|-|-|
|LiftingNet|<center>256<sup>2</sup></center>|85.29|82.75|75.89|73.95|
|MIPLCNet|<center>256<sup>2</sup></center>|86.57|83.38|77.71|75.00|
|ResNet-APReLU|<center>256<sup>2</sup></center>|86.86|84.04|77.89|75.56|
|LSTM-RDRN|<center>256<sup>2</sup></center>|87.71|85.42|**78.50**|76.72|
|BCNN|<center>256<sup>2</sup></center>|81.71|78.63|72.25|70.03|
|**HKG-ResNet34**|<center>256<sup>2</sup></center>|**89.71**|90.54|78.02|**76.74**|
|**HKG-Swin-B-4-12**|<center>256<sup>2</sup></center>|89.57|**92.35**|77.88|76.66|

**Table 2: Results of different evaluation metrics on Cavitation-Long.**
|Methods|<center>Image size|<center>Acc</center>|<center>Pre</center>|<center>Rec</center>|<center>F1</center>|
|-|-|-|-|-|-|
|LiftingNet|<center>256<sup>2</sup></center>|88.43|86.31|87.97|87.05|
|MIPLCNet|<center>256<sup>2</sup></center>|89.14|87.06|88.18|87.55|
|ResNet-APReLU|<center>256<sup>2</sup></center>|90.71|88.83|90.91|89.70|
|LSTM-RDRN|<center>256<sup>2</sup></center>|91.14|89.36|91.55|90.32|
|BCNN|<center>256<sup>2</sup></center>|85.71|82.42|85.09|83.56|
|**HKG-DeneseNet169**|<center>256<sup>2</sup></center>|92.69|88.85|92.48|90.46|
|**HKG-Swin-B-4-12** |<center>256<sup>2</sup></center>|**93.18**|**89.58**|**93.40**|**91.27**|

**Table 2: Results of different evaluation metrics on Cavitation-Noise.**
|Methods|<center>Image size|<center>Acc</center>|<center>Pre</center>|<center>Rec</center>|<center>F1</center>|
|-|-|-|-|-|-|
|LiftingNet|<center>256<sup>2</sup></center>|95.86|94.62|95.84|95.20|
|MIPLCNet|<center>256<sup>2</sup></center>|96.57|95.78|96.61|96.18|
|ResNet-APReLU|<center>256<sup>2</sup></center>|97.29|96.77|97.49|97.12|
|LSTM-RDRN|<center>256<sup>2</sup></center>|98.71|98.40|98.53|98.46|
|BCNN|<center>256<sup>2</sup></center>|92.14|89.75|92.71|91.03|
|**HKG-DeneseNet169**|<center>256<sup>2</sup></center>|99.25|99.25|99.25|99.25|
|**HKG-Swin-B-4-12** |<center>256<sup>2</sup></center>|**99.63**|**99.63**|**99.63**|**99.62**|

For the public PUB dataset, we considered various methods proposed by other researchers under different backbone architectures, such as LeNet, AlexNet, ResNet and Transformer.

Furthermore, we conducted ablation studies to investigate the impact of different factors such as word embedding methods, correlation matrices, GCN layers, parameter sensitivity, STFT parameters, window sizes, and down-sampling on our method to ensure its robustness and effectiveness.

In conclusion, our designed experiments are reasonable and can verify the effectiveness of our method.


2.&nbsp;**Presentation:** Minor readability issues.

**Response:** Thank you for your feedback. We have carefully reviewed the paper and will further improve the readability and ensure clarity throughout the text. 


3.&nbsp;**Reproducibility:** Insufficient information for reproducibility.

**Response:** We apologize that our paper does not clearly set out the experimental details in the main paper. As you mentioned, the reproducibility of experiments is crucial. However, we believe that the details provided in the “B.2 Implementation Details” section of the Appendix can support the reproducibility of our method (see [Figure 1](https://gitee.com/SmallStupidFish/HKG/blob/main/Response/Fig1.png)). In the “B.2 Implementation Details” section, we provide detailed information about our method, including data preprocessing, the number of graph network nodes, loss functions, parameter settings, and learning rates. In addition, we offer pseudocode for our method (see [Figure 2](https://gitee.com/SmallStupidFish/HKG/blob/main/Response/Fig2.png)) in the “B.2 Implementation Details” section. Furthermore, we provide a link to the source code of our method (https://github.com/CavitationDetection/HKG), which contains various source code files and test data.


4.&nbsp;**Comment:** Pre-defined nodes & Model relationship & Implicit way

**Response:** Thank you for your insightful comments. We appreciate the opportunity to clarify the novelty and effectiveness of our proposed framework, an end-to-end hierarchical knowledge guided fault intensity diagnosis framework (HKG) inspired by the tree of thought concept, which is amenable to any representation learning methods. 

Our framework distinguishes itself by explicitly incorporating hierarchical relationships among fault classes into the model, using a graph representation to capture these relationships. This hierarchical graph is then integrated within various deep learning architectures (e.g., CNN, LNN and Transformer), to influence the feature extraction process directly. In addition, it is widely recognized that a tree is a specific type of graph. Therefore, we argue that our method is not a traditional directed mapping from input to output, but rather a structured, knowledge-guided methodology based on hierarchical tree representation.

Just as you mentioned, the nodes in our hierarchical label tree need to be predefined. This necessity stems from the considered supervised learning tasks, where knowledge of the dataset’s classes and their interrelations is assumed beforehand, enabling us to delineate the hierarchical structure. Such a delineation is not extensively exploited in existing fault diagnosis methodologies, which our research seeks to amend by infusing hierarchical class knowledge into the representation learning process. This strategy aims to guide or constrain the feature extraction capabilities of deep learning models, ensuring that the extracted features are not only statistically relevant but also hierarchically coherent. In general, any guiding knowledge of the physical world is a priori or predefined by humans. The most important point is that the hierarchical knowledge among classes in our hierarchical label tree is only used to correct the initially established statistical correlation matrix (SCM). This rigid embedding of knowledge is to ensure that the hierarchical knowledge of classes can effectively guide the features extracted by deep learning. Therefore, defining hierarchical nodes in advance is feasible in such tasks.

In addition, we use the word-embedding approach to model the semantic structure of classes, which inherently encapsulates the hierarchical structure between classes. Of course, this is an implicit representation of hierarchical knowledge. Moreover, this method of constructing class information has been proven to be feasible in many research [6, 7, 8, 9]. 

As far as we know, there are currently two main approaches to modelling class hierarchical knowledge: explicit and implicit methods. The implicit methods mainly include label embedding and hierarchical loss. In our study, we used both explicit modelling and the label embedding approach from implicit methods. In summary, our approach can simulate the hierarchical relationships between classes or nodes.


5.&nbsp;**Comment:** Will there be so many classes as nodes?

**Response:** Thank the reviewer for pointing this out. In our method, each sub-node of the hierarchical label tree corresponds to real and specific subclasses, while their parent nodes represent more generalized classes during the transformation of multi-label tasks (see [Figure 3](https://gitee.com/SmallStupidFish/HKG/blob/main/Response/Fig3.png)). Therefore, there may be many class nodes in the hierarchical label tree. It depends on the complexity of the dataset and the number of classes. Different datasets correspond to different hierarchical label trees. In addition, even for the same dataset, the hierarchical label trees may vary depending on the construction rules (see [Figure 3a and 3b](https://gitee.com/SmallStupidFish/HKG/blob/main/Response/Fig3.png)). Our method demonstrates greater advantages with larger and more complex datasets. In practical applications, we can design and construct hierarchical label trees according to specific requirements and data structures.


6.&nbsp;**Comment:** Is it necessary to conduct GCN for relation modeling?

**Response:** Thank you for the comment. We understand your concerns and confusion. We have carefully addressed the above questions after through consideration. We believe that utilizing GCN for modeling hierarchical knowledge among classes is essential and has the following important advantages: Firstly, GCN can effectively captures the nonlinear relationships among classes in complex datasets, particularly when intricate relationships exist within hierarchical label trees (like, ImageNet, ANIMAL-10N and Pathology). Secondly, GCN utilizes the structural information among classes to enhance the model’s understanding and modeling capability of class relationships. In addition, GCN demonstrates adaptability by dynamically adjusting model parameters based on dataset features, thereby better accommodating various task requirements and data structures. In our approach, both Re-HKCM and label embedding are used as inputs to GCN, combining explicit and implicit hierarchical knowledge.

To the best of our knowledge, current hierarchical classification based on deep learning can be broadly classified into three categories: label embedding, hierarchical loss and hierarchical structures. For label embedding methods, the hierarchical information across labels is mapped into hierarchical semantic vectors containing relative position information. For hierarchical losses, which emphasizes the consistency between the prediction results and the class hierarchy that is often employed in multi-label classification tasks. Hierarchical structures aim to better adapt specifically designed deep neural network structures to the class hierarchy of a particular task. In our work, our proposed method is a combination of explicit hierarchical knowledge embedding and implicit label embedding.

For current research, a common approach in label embedding is to use GCN for modelling, which has been validated in many domains and proved to be an effective modelling method [10, 11, 12, 13]. Our proposed approach is generic and particularly suitable for complex datasets. In this case, the advantages of GCN modelling are more prominent as it is better able to capture the complex relationships between classes, thus improving the performance and generalization of the model.

In future work, we will explore learnable hierarchical loss to embed and learn soft hierarchical knowledge, enhancing the generalization capability of our framework.


7.&nbsp;**Comment:** How to set the threshold in equation 8?

**Response:** Thanks for your comments. The Equation (8) is used to address two drawbacks with the Hierarchical Knowledge Correlation Matrix (HKCM): the pattern of co-occurrence between class and other classes may suffer from a long-tailed distribution, and the absolute number of training and testing datasets may not be exactly equivalent. In Equation (8), we use a threshold to convert the HKCM into a binary matrix with values of either 0 or 1. The threshold value ranges from (0,1]. In our ablation experiments, we experimented with threshold of 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, and 1.0, respectively, to determine the most suitable threshold for a specific dataset. The experimental results showed that a threshold of 0.3 produced the best performance on three real-world cavitation datasets and a threshold of 0.4 achieved the best performance on the PUB dataset. Therefore, the optimal threshold varies for different datasets due to the size of the dataset and embedded hierarchical knowledge.

8.&nbsp;**Comment:** The figure 5-(b) is hard to read.

**Response:** Thank you for highlighting the readability issue with Figure 5-(b). We recognize that the inclusion of two axes within a single figure may have compromised its clarity. To address this concern, we are committed to revising the figure to enhance its readability. This will involve either simplifying the presentation or providing a more detailed explanation in the figure caption to ensure that the figure is accessible and understandable to all readers. We value your feedback as it aids in our pursuit of clear and effective communication through our visuals.

Sincerely,

All authors

**References**

[1] Pan J, Zi Y, Chen J, et al. LiftingNet: A novel deep learning network with layerwise feature learning from noisy mechanical data for fault classification[J]. IEEE Transactions on Industrial Electronics, 2017, 65(6): 4973-4982.

[2] Pan T, Chen J, Zhou Z, et al. A novel deep learning network via multiscale inner product with locally connected feature extraction for intelligent fault detection[J]. IEEE Transactions on Industrial Informatics, 2019, 15(9): 5119-5128.

[3] Zhao M, Zhong S, Fu X, et al. Deep residual networks with adaptively parametric rectifier linear units for fault diagnosis[J]. IEEE transactions on industrial electronics, 2020, 68(3): 2587-2597.

[4] Mohammad-Alikhani A, Nahid-Mobarakeh B, Hsieh M F. One-dimensional LSTM-regulated deep residual network for data-driven fault detection in electric machines[J]. IEEE Transactions on Industrial Electronics, 2023.

[5] Zhu X, Bain M. B-CNN: branch convolutional neural network for hierarchical classification[J]. arXiv preprint arXiv:1709.09890, 2017.

[6] Akata Z, Perronnin F, Harchaoui Z, et al. Label-embedding for image classification[J]. IEEE transactions on pattern analysis and machine intelligence, 2015, 38(7): 1425-1438.

[7] Akata Z, Perronnin F, Harchaoui Z, et al. Label-embedding for attribute-based classification[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2013: 819-826.

[8] Kumar V, Pujari A K, Padmanabhan V, et al. Multi-label classification using hierarchical embedding[J]. Expert Systems with Applications, 2018, 91: 263-269.

[9] Chen C, Wang H, Liu W, et al. Two-stage label embedding via neural factorization machine for multi-label classification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 3304-3311.

[10] You R, Guo Z, Cui L, et al. Cross-modality attention with semantic graph embedding for multi-label classification[C]//Proceedings of the AAAI conference on artificial intelligence. 2020, 34(07): 12709-12716.

[11] Chen T, Lin L, Chen R, et al. Knowledge-guided multi-label few-shot learning for general image recognition[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020, 44(3): 1371-1384.

[12] Lanchantin J, Wang T, Ordonez V, et al. General multi-label image classification with transformers[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 16478-16488.

[13] Lan H, Zhu Q, Guan J, et al. Hierarchical Metadata Information Constrained Self-Supervised Learning for Anomalous Sound Detection under Domain Shift[C]//ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024: 7670-7674.

