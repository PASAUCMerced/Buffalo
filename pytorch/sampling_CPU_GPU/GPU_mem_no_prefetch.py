import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse
from statistics import mean
import random
import numpy as np
import time
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../micro_batch_train')
sys.path.insert(0,'../models')
sys.path.insert(0,'../utils')
from memory_usage import see_memory_usage, nvidia_smi_usage

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.mode != 'cpu':
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    # def inference(self, g, device, batch_size):
    #     """Conduct layer-wise inference to get all the node embeddings."""
    #     feat = g.ndata['feat']
    #     sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
    #     dataloader = DataLoader(
    #             g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
    #             batch_size=batch_size, shuffle=False, drop_last=False,
    #             num_workers=0)
    #     buffer_device = torch.device('cpu')
    #     pin_memory = (buffer_device != device)

    #     for l, layer in enumerate(self.layers):
    #         y = torch.empty(
    #             g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
    #             device=buffer_device, pin_memory=pin_memory)
    #         feat = feat.to(device)
    #         for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
    #             x = feat[input_nodes]
    #             h = layer(blocks[0], x) # len(blocks) = 1
    #             if l != len(self.layers) - 1:
    #                 h = F.relu(h)
    #                 h = self.dropout(h)
    #             # by design, our output nodes are contiguous
    #             y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
    #         feat = y
    #     return y
    
# def evaluate(model, graph, dataloader):
#     model.eval()
#     ys = []
#     y_hats = []
#     for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
#         with torch.no_grad():
#             x = blocks[0].srcdata['feat']
#             ys.append(blocks[-1].dstdata['label'])
#             y_hats.append(model(blocks, x))
#     return MF.accuracy(torch.cat(y_hats), torch.cat(ys))

# def layerwise_infer(device, graph, nid, model, batch_size):
#     model.eval()
#     with torch.no_grad():
#         pred = model.inference(graph, device, batch_size) # pred in buffer_device
#         pred = pred[nid]
#         label = graph.ndata['label'][nid].to(pred.device)
#         return MF.accuracy(pred, label)

def train(args, device, g, dataset, model):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    full_batch_size = len(train_idx)
    
    sampler = NeighborSampler([10, 25, 30],  # fanout for [layer-0, layer-1, layer-2]
                            )
    use_uva = (args.mode == 'mixed')
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=full_batch_size, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)

    # val_dataloader = DataLoader(g, val_idx, sampler, device=device,
    #                             batch_size=1024, shuffle=True,
    #                             drop_last=False, num_workers=0,
    #                             use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    train_time_list=[]
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        train_start= time.time()
        see_memory_usage("----------------------------------------before train dataloader ")
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            see_memory_usage("----------------------------------------after train dataloader ")
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            see_memory_usage("----------------------------------------before model ")
            y_hat = model(blocks, x)
            see_memory_usage("----------------------------------------after model ")
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            see_memory_usage("----------------------------------------after backward ")
            opt.step()
            see_memory_usage("----------------------------------------after optimizer.step ")
            total_loss += loss.item()
        train_end= time.time()
        if epoch >= args.log_indent:
                t0 = train_end-train_start
                train_time_list.append(t0)
        print()
        print("Epoch {:05d} | Loss {:4f} ".format(epoch, total_loss))
        # acc = evaluate(model, g, val_dataloader)
        # print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
        #       .format(epoch, total_loss / (it+1), acc.item()))
    print()
    print('mean training time/epoch {}'.format(mean(train_time_list)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # argparser.add_argument("--mode", default='cpu', choices=['cpu', 'mixed', 'puregpu'],
    #                     help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
    #                         "'puregpu' for pure-GPU training.")
    # argparser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
    #                     help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
    #                         "'puregpu' for pure-GPU training.")
    argparser.add_argument("--mode", default='puregpu', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                            "'puregpu' for pure-GPU training.")
    argparser.add_argument('--seed', type=int, default=1236)
    argparser.add_argument('--setseed', type=bool, default=True)
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument("--epochs", type=int, default=20)
    argparser.add_argument('--log-indent', type=int, default=10)
    

    args = argparser.parse_args()
    if args.setseed:
        set_seed(args)
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    # load and preprocess dataset
    print('Loading data')
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
    g = dataset[0]
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, args.num_hidden, out_size).to(device)

    # model training
    print('Training...')
    
    train(args, device, g, dataset, model)
    

    # # test the model
    # print('Testing...')
    # acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    # print("Test Accuracy {:.4f}".format(acc.item()))