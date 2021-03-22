# CS224W Course Project 
Authors: Yichen Yang, Lingjue Xie, Fangchen Li

Dataset:
OGB ogbl-ddi https://ogb.stanford.edu/docs/linkprop/#ogbl-ddi

Method:
DEA https://arxiv.org/abs/2009.00142 + 
JKNet https://arxiv.org/abs/1806.03536

Result:
3-layer DEA-GCN-JK(max pooling)
10 runs (seed 1 to 10)
Val Hits@20: 0.6713 ± 0.0071
Test Hits@20: 0.7672 ± 0.0265

Please see the report for performance of other models and hyperparameters.

Setup:
Simply run the notebook on Google Colab :) 
