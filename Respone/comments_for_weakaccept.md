Dear Reviewer,

Thank you for your detailed review and insightful comments on our work. We really appreciate your through assessment of the proposed hierarchical knowledge guided fault intensity diagnosis (HKG) and your constructive feedbacks.

1.&nbsp;**Comment:** The proposed method is subject to domain shifting, as the proposed Re-HKCM is based on statistical correlation matrix.

**Response:** Thank you for highlighting this concern. In our approach, domain shifting is indeed an important consideration. We can discuss this issue from different perspectives.

From the perspective of different datasets, domain shifting exists. In our study, the statistical correlation matrix (SCM) is built based on the co-occurrence patterns among classes of datasets, which makes the SCM dependent on the dataset and its size is controlled by the number of classes. Meanwhile, the hierarchical knowledge correlation matrix (HKCM) embeds the hierarchical edge knowledge between different classes into the SCM, where the hierarchical knowledge among classes is dependent on the classes of the dataset and different classes of the dataset lead to different hierarchical knowledge. As a result, the HKCM is affected by the domain shifting. However, our method shows good performance on three real-world cavitation datasets (Cavitation-Short, Cavitation-Long and Cavitation-Noise) and a public bearing dataset (PUB), where the hierarchical structure of the PUB dataset (see Figure A3 in the appendix of the paper) is more complex than the hierarchical structure of the three cavitation datasets. It indicates that our method overcomes a certain degree of domain shifting due to the following rigorous preprocessing techniques of binary transformation (B-HKCM) and re-weighted (Re-HKCM).

But from a general perspective, we think that the domain shifting is actually not a concern. In our study, the proposed method is within supervised learning framework, where we know the hierarchical structure between different classes of the specific dataset. In general, any knowledge of the physical world is a priori from human. At the same time, the classes of different datasets are transformed by word-embedding to have hierarchical semantic structure in the vector space. Of course, the larger and more complex the dataset is, the more complex hierarchical knowledge it corresponds to. However, their hierarchical knowledge can all be represented by a hierarchical tree or a graph, and they all satisfy the more general representation of Equation 6 in the manuscript.

In addition, in our future work, we will explore the hierarchical losses of learning from dataset itself, where effects of domain shifting will not be there at all and which also can enhance the generalization capabilities of our framework.

2.&nbsp;**Comment:** CoT and ToT in Fig. 1.

**Response:** Thank you for pointing out this misconception. We apologize for any confusion caused by the analogy in Fig. 1. As you mentioned, both Chain-of-Thought (CoT) and Tree-of-Thought (ToT) are commonly used in LLM-based reasoning tasks. In our study, we consider convolutional neural network (CNN) as a form of direct mapping from input to output without considering any guiding knowledge, akin to the CoT. However, in our approach, we explicitly consider the hierarchical relationships among classes and represent them using a graph. They are embedded in different deep learning frameworks (e.g. CNN, LNN and Transformer) to guide the extracted hiden features. In addition, it is widely recognized that a tree is a specific type of graph. Therefore, we argue that our method is not a traditional directed mapping from input to output, but rather a knowledge-guided approach based on hierarchical tree representation. Based on these considerations, it is appropriate to describe our approach as a ToT. Furthermore, the metaphorical comparison of our method to a ToT is intended to better illustrate our hierarchical knowledge-guided process, rather than directly comparing it to CoT and ToT in LLM-based reasoning tasks.

3.&nbsp;**Comment:** Figure 5 and Figure A2 with some problems.

**Response:**  We appreciate your feedback regarding the readability of the figures. We accordingly adjust the size of Figure 5 and Figure A2 to improve their visibility. In addition, we will reconsider the necessity of including a bar plot inside another plot and simplify the visualization if needed. 

4.&nbsp;**Comment:** Random seed & cross-validation & single run & train-test data splits

**Response:** We apologize for the oversight in not providing details about random seed setting and cross-validation. Firstly, we split three real-world cavitation datasets (Cavitation-Short, Cavitation-Long and Cavitation-Noise) and one public industrial bearing dataset (PUB) using different random seed, with 80% for training and 20% for testing. This operation eliminates variations in model performance due to the randomness of the training process and data partitioning. In addition, each of our experimental results is an average of three runs to mitigate the impact of random fluctuations and ensure the reliability of our results. Finally, we will add these details in the Experimental Details section in the Appendix.

5.&nbsp;**Comment:** How to handle domain shifting between different datasets?

**Response:** Thank you so much for raising this issue. Handling domain shifting between different datasets is a crucial consideration. For our discussion and analysis of this study about domain shifting, see **comment 1** in above. In fact, the proposed Binary HEKCM (B-HEKCM) and Re-weighted HEKCM (Re-HKCM) not only address the issues mentioned in our paper, but also they can alleviate the problem of domain shifting. The specific additional roles of binarization and re-weighting are as follows:

- Binarization: By binarizing data or features, the model can focus more on the commonalities rather than differences between different domains.
- Re-weighting: By adjusting the weights of different domains to balance domain shifting, which can help the model better adapt to differences between different domains and improves the model’s generalization ability. In general, domain with larger domain shifting is assigned smaller weights, while those with smaller domain shifting is assigned larger weights.

While binarization and re-weighting can alleviate the problem of domain shifting, they cannot fundamentally address this problem. After conducting a literature review and considering our current approach, we have contemplated methods to address domain shifting, as follows:
- Hierarchical Losses: Hierarchical loss is particularly suitable for handling data with a hierarchical structure. The design of the loss function considers the hierarchical relationships of the data, which helps the model better learn and utilize hierarchical information. In addition, it is a soft hierarchical knowledge to guide deep features and learn from the data itself, which improves understanding and generalization of the task. 
- Model Adaptation Strategy: Adjusting our algorithm through model adaptation strategies enables it to perform well in different datasets and domains. This includes tuning model parameters and fine-tuning on different domain data.
- Domain Adaptation Methods: Introducing domain adaptation loss (e.g. Adversarial Loss, Gradient Reversal Layer Loss, Maximum Mean Discrepancy Loss, Kullback-Leibler Divergence Loss, Conditional Entropy Loss and so on) during training process to reduce differences between domains, which enhances the generalization ability of the model.

The above are some thoughts on combining our method to address domain shifting. We believe that the comprehensive integration of these methods can minimize the impact of domain shifting between different datasets on model performance.


Sincerely,

All authors

