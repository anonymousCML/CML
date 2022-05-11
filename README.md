# CML

PyTorch implementation for paper: Contrastive Meta Learning with Behavior Multiplicity forRecommendation



## Dependencies

- Python 3.6
- torch==1.8.1+cu111
- scipy==1.6.2
- tqdm==4.61.2



## Dataset Preparation

#### Raw data：
- IJCAI contest:  https://tianchi.aliyun.com/dataset/dataDetail?dataId=47
- Retail Rocket: https://www.kaggle.com/retailrocket/ecommerce-dataset
- Tmall:  https://tianchi.aliyun.com/dataset/dataDetail?dataId=649 



## Usage
You need to create the `History/` and the `Models/` directories. 
- mkdir /History
- mkdir /Model 


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


## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{wei2022contrastive,
  title={Contrastive meta learning with behavior multiplicity for recommendation},
  author={Wei, Wei and Huang, Chao and Xia, Lianghao and Xu, Yong and Zhao, Jiashu and Yin, Dawei},
  booktitle={Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
  pages={1120--1128},
  year={2022}
}
```



<!-- ## Acknowledgement
 -->







