import torch
import copy
import numpy as np
import networkx as nx
import random
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
import argparse

from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_networkx
from torch_geometric.nn import GCNConv, SAGEConv, TAGConv, JumpingKnowledge

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

# Shortest Path Distance 
def get_spd_matrix(G, S, max_spd=5):
    spd_matrix = np.zeros((G.number_of_nodes(), max_spd + 1), dtype=np.int32)
    for i, node_S in enumerate(S):
        for node, length in nx.shortest_path_length(G, source=node_S).items():
            spd_matrix[node, min(length, max_spd)] += 1
    return spd_matrix /(len(S))

# Random Walk Landing Probablity (S = all nodes)
def get_lp_matrix(A, max_steps=5):
    W = A / A.sum(1, keepdims=True)
    W_list = [np.identity(A.shape[0])]
    for i in range(max_steps):
        W_list.append(np.matmul(W_list[-1], W))
    W_stack = np.stack(W_list, axis=2)  
    return W_stack.mean(axis=0)

def train(model, optimizer, evaluator, graph, x_feature, edge_index, adj_t, split_idx, device,
          batch_size=1024*64, num_epochs=200, save_model=False):
    best_val_score = 0
    best_epoch = 0
    best_test_score = 0
    best_model = model

    all_pos_edges = split_idx['train']['edge'].transpose(0,1).to(device)

    for epoch in range(1, num_epochs+1):
        sum_loss = 0
        count = 0 
        for batch in DataLoader(list(range(all_pos_edges.shape[1])), batch_size=batch_size, shuffle=True):
            model.train()
            batch_pos_edges = all_pos_edges[:, batch]
            batch_neg_edges = negative_sampling(edge_index=edge_index, 
                                            num_nodes=graph.num_nodes,
                                            num_neg_samples=batch_pos_edges.shape[1], 
                                            method='dense').to(device)
            edge_label_index = torch.cat([batch_pos_edges, batch_neg_edges], dim=1).to(device)
          
            pos_label = torch.ones(batch_pos_edges.shape[1], )
            neg_label = torch.zeros(batch_neg_edges.shape[1], )
            edge_label = torch.cat([pos_label, neg_label], dim=0).to(device)

            optimizer.zero_grad()  
            pred = model(x_feature, adj_t, edge_label_index)
            loss = model.loss(pred, edge_label.type_as(pred))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
            optimizer.step()

            sum_loss += loss.item() * edge_label.shape[0]
            count += edge_label.shape[0]

        val_score, test_score = evaluate(model, x_feature, adj_t, split_idx, evaluator)
        if best_val_score < val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_test_score = test_score
            if save_model:
                best_model = copy.deepcopy(model)

        log = 'Epoch: {:03d}, Loss: {:.4f}, Val Hits: {:.2f}%, Test Hits: {:.2f}%'
        print(log.format(epoch, sum_loss/count, 100*val_score, 100*test_score))

    print('Final model:')
    log = 'Epoch: {:03d}, Val Hits: {:.2f}%, Test Hits: {:.2f}%'
    print(log.format(best_epoch, 100*best_val_score, 100*best_test_score))
    return best_model, best_val_score, best_test_score

@torch.no_grad()
def evaluate(model, x_feature, adj_t, split_idx, evaluator):
    model.eval()

    pos_edge_label_index = split_idx['valid']['edge'].transpose(0,1)
    neg_edge_label_index = split_idx['valid']['edge_neg'].transpose(0,1)
    
    y_pred_pos = model(x_feature, adj_t, pos_edge_label_index)
    y_pred_neg = model(x_feature, adj_t, neg_edge_label_index)
    
    score_val = evaluator.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})['hits@20']

    pos_edge_label_index = split_idx['test']['edge'].transpose(0,1)
    neg_edge_label_index = split_idx['test']['edge_neg'].transpose(0,1)
    
    y_pred_pos = model(x_feature, adj_t, pos_edge_label_index)
    y_pred_neg = model(x_feature, adj_t, neg_edge_label_index)
    
    score_test = evaluator.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})['hits@20']

    return (score_val, score_test)

class DEA_GNN_JK(torch.nn.Module):
    def __init__(self, num_nodes, embed_dim, 
                 gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gnn_num_layers, 
                 mlp_in_dim, mlp_hidden_dim, mlp_out_dim=1, mlp_num_layers=2, 
                 dropout=0.5, gnn_batchnorm=False, mlp_batchnorm=False, K=2, jk_mode='max'):
        super(DEA_GNN_JK, self).__init__()
        
        assert jk_mode in ['max','sum','mean','lstm','cat']
        # Embedding
        self.emb = torch.nn.Embedding(num_nodes, embedding_dim=embed_dim)

        # GNN 
        convs_list = [TAGConv(gnn_in_dim, gnn_hidden_dim, K)]
        for i in range(gnn_num_layers-2):
            convs_list.append(TAGConv(gnn_hidden_dim, gnn_hidden_dim, K))
        convs_list.append(TAGConv(gnn_hidden_dim, gnn_out_dim, K))
        self.convs = torch.nn.ModuleList(convs_list)

        # MLP
        lins_list = [torch.nn.Linear(mlp_in_dim, mlp_hidden_dim)]
        for i in range(mlp_num_layers-2):
            lins_list.append(torch.nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
        lins_list.append(torch.nn.Linear(mlp_hidden_dim, mlp_out_dim))
        self.lins = torch.nn.ModuleList(lins_list)

        # Batchnorm
        self.gnn_batchnorm = gnn_batchnorm
        self.mlp_batchnorm = mlp_batchnorm
        if self.gnn_batchnorm:
            self.gnn_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(gnn_hidden_dim) for i in range(gnn_num_layers)])
        
        if self.mlp_batchnorm:
            self.mlp_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(mlp_hidden_dim) for i in range(mlp_num_layers-1)])

        self.jk_mode = jk_mode
        if self.jk_mode in ['max', 'lstm', 'cat']:
            self.jk = JumpingKnowledge(mode=self.jk_mode, channels=gnn_hidden_dim, num_layers=gnn_num_layers)

        self.dropout = dropout
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.emb.weight)  
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        if self.gnn_batchnorm:
            for bn in self.gnn_bns:
                bn.reset_parameters()
        if self.mlp_batchnorm:
            for bn in self.mlp_bns:
                bn.reset_parameters()
        if self.jk_mode in ['max', 'lstm', 'cat']:
            self.jk.reset_parameters()

    def forward(self, x_feature, adj_t, edge_label_index):
        
        if x_feature is not None:
            out = torch.cat([self.emb.weight, x_feature], dim=1)
        else:
            out = self.emb.weight

        out_list = []
        for i in range(len(self.convs)):
            out = self.convs[i](out, adj_t)
            if self.gnn_batchnorm:
                out = self.gnn_bns[i](out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            out_list += [out]

        if self.jk_mode in ['max', 'lstm', 'cat']:
            out = self.jk(out_list)
        elif self.jk_mode == 'mean':
            out_stack = torch.stack(out_list, dim=0)
            out = torch.mean(out_stack, dim=0)
        elif self.jk_mode == 'sum':
            out_stack = torch.stack(out_list, dim=0)
            out = torch.sum(out_stack, dim=0)

        gnn_embed = out[edge_label_index,:]
        embed_product = gnn_embed[0, :, :] * gnn_embed[1, :, :]
        out = embed_product

        for i in range(len(self.lins)-1):
            out = self.lins[i](out)
            if self.mlp_batchnorm:
                out = self.mlp_bns[i](out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lins[-1](out).squeeze(1)

        return out
    
    def loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DEA-JKNet on ogbl-ddi')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--jk_mode', type=str, default="max",
                        help='JKNet aggregation method [max, mean, lstm, sum, cat] (default: max)')
    parser.add_argument('--remove_batchnorm', action='store_true',
                        help='remove batchnorm layers')
    parser.add_argument('--use_stored_x', action='store_true',
                        help='use precomputed distance encoding and node statistics')

    parser.add_argument('--embed_dim', type=int, default=256,
                        help='initial node embedding dimension')
    parser.add_argument('--gnn_hidden_dim', type=int, default=256,
                        help='GNN(DEA) hidden layer dimension (default: 256)')
    parser.add_argument('--gnn_num_layers', type=int, default=3,
                        help='number of GNN(DEA) message passing layers (default: 3, must be greater than 1)')
    parser.add_argument('--k', type=int, default=2,
                        help='GNN(DEA) number of hops')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                        help='linear hidden layer dimension for edge prediction (default: 256)')
    parser.add_argument('--mlp_num_layers', type=int, default=2,
                        help='number of linear layers for edge prediction (default: 2, must be greater than 1)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=1024*64,
                        help='input batch size for training (default: 1024*64)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--runs', type=int, default=10,
                        help='number of runs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Adam learning rate (default: 0.005')

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print('Device: {}'.format(device))

    dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    graph = dataset[0]
    split_idx = dataset.get_edge_split()
    evaluator = Evaluator(name='ogbl-ddi')
    edge_index = graph.edge_index
    adj_t = torch_sparse.SparseTensor.from_edge_index(edge_index)

    print('Train edges:', split_idx['train']['edge'].shape)
    print('Val edges:', split_idx['valid']['edge'].shape)
    print('Val negative edges:', split_idx['valid']['edge_neg'].shape)
    print('Test edges:', split_idx['test']['edge'].shape)
    print('Test negative edges:', split_idx['test']['edge_neg'].shape)

    if not args.use_stored_x:
        print('start computing extra features (~30 min):')
        nx_graph = to_networkx(graph, to_undirected=True)
        nx_degree = nx.degree(nx_graph)
        nx_pagerank = nx.pagerank(nx_graph)

        nx_clustering = nx.clustering(nx_graph)
        nx_centrality = nx.closeness_centrality(nx_graph)

        # S=200 nodes
        # np.random.seed(0)
        node_subset = np.random.choice(nx_graph.number_of_nodes(), size=200, replace=False)
        spd_feature = get_spd_matrix(G=nx_graph, S=node_subset, max_spd=5)

        # S = all nodes
        lp_feature = get_lp_matrix(adj_t.to_dense(), max_steps=5)

        # Convert to tensor
        tensor_degree = torch.Tensor([t[1] for t in nx_degree]).unsqueeze(1)
        tensor_pagerank = torch.Tensor([t[1] for t in nx_pagerank.items()]).unsqueeze(1)
        tensor_clustering = torch.Tensor([t[1] for t in nx_clustering.items()]).unsqueeze(1)
        tensor_centrality = torch.Tensor([t[1] for t in nx_centrality.items()]).unsqueeze(1)

        tensor_spd = torch.Tensor(spd_feature)
        tensor_lp = torch.Tensor(lp_feature)

        # Concat
        feature_tensor_list = [tensor_degree, tensor_pagerank, tensor_clustering,
                               tensor_centrality, tensor_spd, tensor_lp]
        x_feature = torch.cat(feature_tensor_list, dim=1)
        print('extra feature shape:', x_feature.shape)
    else:
        print('load extra features:')
        x_df = pd.read_csv('x_feature.csv')
        x_feature_numpy = x_df.to_numpy()
        x_feature = torch.Tensor(x_feature_numpy)
        print('extra feature shape:', x_feature.shape)

    # Normalize to 0-1
    x_max = torch.max(x_feature, dim=0, keepdim=True)[0]
    x_min = torch.min(x_feature, dim=0, keepdim=True)[0]
    x_feature = (x_feature - x_min) / (x_max - x_min + 1e-6)

    edge_index = edge_index.to(device)
    adj_t = adj_t.to(device)
    x_feature = x_feature.to(device)

    gnn_in_dim = args.embed_dim + x_feature.shape[1]

    if args.jk_mode == 'cat':
        mlp_in_dim = args.gnn_hidden_dim * args.gnn_num_layers
    else:
        mlp_in_dim = args.gnn_hidden_dim
    model = DEA_GNN_JK(num_nodes=graph.num_nodes, embed_dim=args.embed_dim,
                       gnn_in_dim=gnn_in_dim, gnn_hidden_dim=args.gnn_hidden_dim, gnn_out_dim=args.gnn_hidden_dim,
                       gnn_num_layers=args.gnn_num_layers, mlp_in_dim=mlp_in_dim, mlp_hidden_dim=args.mlp_hidden_dim,
                       mlp_out_dim=1, mlp_num_layers=args.mlp_num_layers,
                       dropout=args.dropout, gnn_batchnorm=not args.remove_batchnorm,
                       mlp_batchnorm=not args.remove_batchnorm,
                       K=args.k, jk_mode=args.jk_mode).to(device)

    print(model)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    # Multiple runs
    RUNS = args.runs
    best_val_scores = np.zeros((RUNS,))
    best_test_scores = np.zeros((RUNS,))

    for i in range(RUNS):
        random.seed(i + 1)
        torch.manual_seed(i + 1)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        result = train(model, optimizer, evaluator, graph, x_feature, edge_index, adj_t, split_idx,
                       device=device, batch_size=args.batch_size, num_epochs=args.epochs, save_model=False)

        best_val_scores[i] = result[1]
        best_test_scores[i] = result[2]

        print('Run', i + 1, 'done.')

    log = 'Mean Val Hits: {:.4f}, SD Val Hits: {:.4f}'
    print(log.format(np.mean(best_val_scores), np.std(best_val_scores, ddof=1)))
    log = 'Mean Test Hits: {:.4f}, SD Test Hits: {:.4f}'
    print(log.format(np.mean(best_test_scores), np.std(best_test_scores, ddof=1)))


if __name__ == "__main__":
    main()