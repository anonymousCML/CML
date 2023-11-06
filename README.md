# CML
 


This repository contains PyTorch codes and datasets for the paper:

> Wei, Wei and Huang, Chao and Xia, Lianghao and Xu, Yong and Zhao, Jiashu and Yin, Dawei. Contrastive Meta Learning with Behavior Multiplicity forRecommendation. <a href='https://arxiv.org/pdf/2202.08523.pdf'>Paper in arXiv</a>.


## Introduction
Contrastive Meta Learning (CML) leverages multi-behavior learning paradigm to model diverse and multiplex user-item relationships, as well as tackling the label scarcity problem for target behaviors. The designed multi-behavior contrastive task is to capture the transferable user-item relationships from multi-typed user behavior data heterogeneity. And the proposed meta contrastive encoding scheme allows CML to preserve the personalized multi-behavior characteristics, so as to be reflective of the diverse behavior-aware user preference under a customized self-supervised framework.


## Citation
```
@inproceedings{wei2022contrastive,
  title={Contrastive meta learning with behavior multiplicity for recommendation},
  author={Wei, Wei and Huang, Chao and Xia, Lianghao and Xu, Yong and Zhao, Jiashu and Yin, Dawei},
  booktitle={Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
  pages={1120--1128},
  year={2022}
}
```


## Environment

The codes of CML are implemented and tested under the following development environment:

- Python 3.6
- torch==1.8.1+cu111
- scipy==1.6.2
- tqdm==4.61.2



## Datasets

#### Raw data：
- IJCAI contest:  https://tianchi.aliyun.com/dataset/dataDetail?dataId=47
- Retail Rocket: https://www.kaggle.com/retailrocket/ecommerce-dataset
- Tmall:  https://tianchi.aliyun.com/dataset/dataDetail?dataId=649 
#### Processed data：
- The processed IJCAI are under the /datasets folder.


## Usage

The command to train CML on the Tmall/IJCAI/Retailrocket datasets are as follows. The commands specify the hyperparameter settings that generate the reported results in the paper.

* Tmall
```
python .\main.py --dataset=Tmall --opt_base_lr=1e-3 --opt_max_lr=5e-3 --opt_weight_decay=1e-4 --meta_opt_base_lr=1e-4 --meta_opt_max_lr=2e-3 --meta_opt_weight_decay=1e-4 --meta_lr=1e-3 --batch=8192 --meta_batch=128 --SSL_batch=18
```
* IJCAI
```
python .\main.py --dataset=IJCAI_15 --sampNum=10 --opt_base_lr=1e-3 --opt_max_lr=2e-3 --opt_weight_decay=1e-4 --meta_opt_base_lr=1e-4 --meta_opt_max_lr=1e-3 --meta_opt_weight_decay=1e-4 --meta_lr=1e-3 --batch=8192 --meta_batch=128 --SSL_batch=30 
```
* Retailrocket
```
python .\main.py --dataset='retailrocket' --sampNum=40 --lr=3e-4 --opt_base_lr=1e-4 --opt_max_lr=1e-3 --opt_weight_decay=1e-4 --opt_weight_decay=1e-4 --meta_opt_base_lr=1e-4 --meta_opt_max_lr=1e-3 --meta_opt_weight_decay=1e-3 --meta_lr=1e-3 --batch=2048 --meta_batch=128 --SSL_batch=15
```         
        


<!-- Important arguments:
* `reg`: It is the weight for weight-decay regularization. We tune this hyperparameter from the set `{1e-2, 1e-3, 1e-4, 1e-5}`.
* `ssl_reg`: This is the weight for the hypergraph-graph contrastive learning loss. The value is tuned from `1e-2` to `1e-8`.
* `temp`: This is the temperature factor in the InfoNCE loss in our contrastive learning. The value is selected from `{10, 3, 1, 0.3, 0.1}`.
* `keepRate`: It denotes the rate to keep edges in the graph dropout, which is tuned from `{0.25, 0.5, 0.75, 1.0}`.
 -->





<!-- ## Acknowledgement
 -->





It will be released again in few days in the optimized code version.







