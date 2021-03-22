# CS224W Course Project  
This is a course project for [Stanford CS224W](http://web.stanford.edu/class/cs224w/) (Machine Learning with Graphs) winter 2021, which achieved SOTA on OGB leaderboard.  
## Authors: 
Yichen Yang, Lingjue Xie, Fangchen Li  

## Dataset:  
OGB ogbl-ddi https://ogb.stanford.edu/docs/linkprop/#ogbl-ddi  

## Method:   
DEA https://arxiv.org/abs/2009.00142 +   
JKNet https://arxiv.org/abs/1806.03536  

## Setup: 
```
ogb>=1.3.0
torch>=1.8.0
```
For ipynb, simply run the notebook on Google Colab :)  
  
For py,
first install requirements:
```
pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install -q torch-geometric
pip install ogb
```

To replicate result on the leaderboard, use
```
python dea_gcn_jk.py --use_stored_x
```
this gives result of DEA-JK-Max (3 layers)  

Important commandline arguments
- `jk_mode`: how JKNet aggregates multiple layers [max, mean, lstm, sum, cat]
- `k`: number of hops for each gcn layer (DEA-GCN)
- `gnn_num_layers`: number of DEA-GCN layers
- `use_stored_x`: use precomputed distance encoding(DE) and node statistics to save time

for other arguments, please check
```
python dea_gcn_jk.py --help
```

for example,
```
python dea_gcn_jk.py --use_stored_x --gnn_num_layers 2 --k 3 --jk_mode mean
```

## Result:   

| Model       | Val Hits@20     | Test Hits@20    | Parameters | Hardware   |
| ------------|-----------------| ----------------|------------|------------|
| DEA-JK-Mean (3 layers) | 0.6885 ± 0.0048 | 0.7133 ± 0.1519 | 1763329    | Tesla P100 |
| DEA-JK-LSTM (3 layers) | 0.6900 ± 0.0039 | 0.7662 ± 0.0681 | 3736322    | Tesla K80  |
| DEA-JK-Max  (3 layers) | 0.6713 ± 0.0071 | 0.7672 ± 0.0265 | 1763329    | Tesla T4   |

\* DEA-JK-Mean and DEA-JK_LSTM use 5 runs, DEA-JK-Max (3 layers) use 10 runs.  
Please see the [report](https://github.com/JeffJeffy/CS224W-OGB-DEA-JK/blob/main/CS224w_final_report.pdf) for discussions of other models and hyperparameters.   

 
