import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../pytorch/utils/')
sys.path.insert(0,'../../pytorch/micro_batch_train/')
sys.path.insert(0,'../../pytorch/models/')

import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from block_dataloader import generate_dataloader_block

import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

import random
from graphsage_model_wo_mem import GraphSAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_cora, load_karate, prepare_data, load_pubmed

from load_graph import load_ogbn_dataset, load_ogb
from memory_usage import see_memory_usage, nvidia_smi_usage
import tracemalloc
from cpu_mem_usage import get_memory
from statistics import mean
# from draw_graph import gen_pyvis_graph_local,gen_pyvis_graph_global,draw_dataloader_blocks_pyvis
# from draw_graph import draw_dataloader_blocks_pyvis_total
from my_utils import parse_results

import pickle
from utils import Logger
import os 
import numpy
from collections import Counter




def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)

def CPU_DELTA_TIME(tic, str1):
	toc = time.time()
	print(str1 + ' spend:  {:.6f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	# train_nid = train_nid.to(device)
	# val_nid=val_nid.to(device)
	# test_nid=test_nid.to(device)
	nfeats=nfeats.to(device)
	g=g.to(device)
	# print('device ', device)
	model.eval()
	with torch.no_grad():
		# pred = model(g=g, x=nfeats)
		pred = model.inference(g, nfeats,  args, device)
	model.train()
	
	train_acc= compute_acc(pred[train_nid], labels[train_nid].to(pred.device))
	val_acc=compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
	test_acc=compute_acc(pred[test_nid], labels[test_nid].to(pred.device))
	return (train_acc, val_acc, test_acc)


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

def load_block_subtensor(nfeat, labels, blocks, device,args):
	"""
	Extracts features and labels for a subset of nodes
	"""

	if args.GPUmem:
		see_memory_usage("----------------------------------------before batch input features to device")
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	if args.GPUmem:
		see_memory_usage("----------------------------------------after batch input features to device")
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	if args.GPUmem:
		see_memory_usage("----------------------------------------after  batch labels to device")
	return batch_inputs, batch_labels

def get_compute_num_nids(blocks):
	res=0
	for b in blocks:
		res+=len(b.srcdata['_ID'])
	return res

	
def get_FL_output_num_nids(blocks):
	
	output_fl =len(blocks[0].dstdata['_ID'])
	return output_fl



#### Entry point
def run(args, device, data):
	if args.GPUmem:
		see_memory_usage("----------------------------------------start of run function ")
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in feats: ', in_feats)
	nvidia_smi_list=[]
	# print('the mode of the raw graph')
	# print(torch.mode(g.in_degrees()))
	# return
	# draw_nx_graph(g)
	# gen_pyvis_graph_global(g,train_nid)
	if args.selection_method =='metis':
		args.o_graph = dgl.node_subgraph(g, train_nid)


	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	full_batch_size = len(train_nid)
	

	args.num_workers = 0
	full_batch_dataloader = dgl.dataloading.DataLoader(
		g,
		train_nid,
		sampler,
		# device='cpu',
		batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	if args.GPUmem:
		see_memory_usage("----------------------------------------before model to device ")


	model = GraphSAGE(
					in_feats,
					args.num_hidden,
					n_classes,
					args.aggre,
					args.num_layers,
					F.relu,
					args.dropout).to(device)
					
	loss_fcn = nn.CrossEntropyLoss()

	if args.GPUmem:
				see_memory_usage("----------------------------------------after model to device")
	logger = Logger(args.num_runs, args)
	dur = []
	time_block_gen=[]
	loading_time_list=[]
	for run in range(args.num_runs):
		model.reset_parameters()
		# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		for epoch in range(args.num_epochs):
			print(' Run '+str(run)+'| Epoch '+ str( epoch)+' |')
			num_src_node =0
			num_out_node_FL=0
			gen_block=0
			tmp_t=0
			model.train()
			if epoch >= args.log_indent:
				t0 = time.time()
			loss_sum=0
			# start of data preprocessing part---s---------s--------s-------------s--------s------------s--------s----
			if args.load_full_batch:
				loading_start_time = time.time()#
				full_batch_dataloader=[]
				file_name=r'./../../../dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
				with open(file_name, 'rb') as handle:
					item=pickle.load(handle)
					full_batch_dataloader.append(item)
				loading_end_time = time.time()#
				loading_time_list.append(loading_end_time-loading_start_time)#
			if args.GPUmem:
				see_memory_usage("----------------------------------------before generate dataloader block ")
			
			block_dataloader, weights_list, time_collection = generate_dataloader_block(g, full_batch_dataloader, args)
			if args.GPUmem:
				see_memory_usage("-----------------------------------------after block dataloader generation ")
			connect_check_time, block_gen_time_total, batch_blocks_gen_time =time_collection
			print('	---Connection check (sec): ', connect_check_time)
			print('	---Block construction (sec) ', block_gen_time_total)
			# print('average batch blocks generation time: ', batch_blocks_gen_time)
			# end of data preprocessing part------e---------e-------e----------e------e----------e--------e-------e--

			if epoch >= args.log_indent:
				gen_block=time.time() - t0
				time_block_gen.append(time.time() - t0)
				# print('block dataloader generation time/epoch {}'.format(np.mean(time_block_gen)))
				tmp_t=time.time()
			# Loop over the dataloader to sample the computation dependency graph as a list of blocks.
			
			pseudo_mini_loss = torch.tensor([], dtype=torch.long)
			data_loading_t=[]
			feature_block_loading=[]
			modeling_t=[]
			loss_cal_t=[]
			backward_t=[]
			data_size_transfer=[]
			blocks_size=[]
			num_input_nids=0
			time_ex=0
			pure_train_time = 0
			tts=time.time()
			for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
				
				tt1=time.time()
				batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)#------------*
				blocks = [block.int().to(device) for block in blocks]#------------*
				tt5=time.time()
				feature_block_loading.append(tt5-tt1)


				# Compute loss and prediction
				batch_pred = model(blocks, batch_inputs)#------------*
				pseudo_mini_loss = loss_fcn(batch_pred, batch_labels)#------------*
				pseudo_mini_loss = pseudo_mini_loss*weights_list[step]#------------*
				pseudo_mini_loss.backward()#------------*

				loss_sum += pseudo_mini_loss#------------*
				pure_train_time += time.time()-tt5
				
			time13 = time.time()
			optimizer.step()
			optimizer.zero_grad()
			ttend=time.time()
			pure_train_time += (ttend-time13)

			if epoch >= args.log_indent:
				tmp_t2=time.time()
				# pure_train_time = ttend-tts-sum(feature_block_loading)
				
				# pure_train_time_list.append(pure_train_time)
				full_epoch=time.time() - t0
				dur.append(time.time() - t0)
				# print('All time are int the unit second')
				
				print('	---Data loading (sec) {}'.format((loading_end_time-loading_start_time)+(sum(feature_block_loading))))
				print('	---Training time on GPU (sec) {}'.format(pure_train_time))
				print('---End to end time (sec) {}'.format(np.mean(dur)))
				# print('Training time without block to device /epoch {}'.format(tmp_t2-tmp_t-time_ex-sum(block_to_t)))
				# # print('Training time without total dataloading part /epoch {}'.format(tmp_t2-tmp_t-sum(block_to_t)-sum(data_loading_t)))
				# print('Training time without total dataloading part /epoch {}'.format(sum(modeling_t)+sum(loss_cal_t)+sum(backward_t)+ttend-tte))
				# print('load block tensor time/epoch {}'.format(np.sum(data_loading_t)))
				# print('block to device time/epoch {}'.format(np.sum(block_to_t)))
				# print('input features size transfer per epoch {}'.format(np.sum(data_size_transfer)/1024/1024/1024))
				# print('blocks size to device per epoch {}'.format(np.sum(blocks_size)/1024/1024/1024))

			# if args.eval:
			
			# 	train_acc, val_acc, test_acc = evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args)

			# 	logger.add_result(run, (train_acc, val_acc, test_acc))
					
			# 	print("Run {:02d} | Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(run, epoch, loss_sum.item(), train_acc, val_acc, test_acc))
			# else:
			# 	print(' Run '+str(run)+'| Epoch '+ str( epoch)+' |')
			# print('Number of nodes for computation during this epoch: ', num_src_node)
			# print('Number of first layer input nodes during this epoch: ', num_input_nids)
			# print('Number of first layer output nodes during this epoch: ', num_out_node_FL)
	# 	if args.eval:
	# 		logger.print_statistics(run)

	# if args.eval:
	# 	logger.print_statistics()
	# print(model)
	# count_parameters(model)
	
# def count_parameters(model):
# 	pytorch_total_params = sum(torch.numel(p) for p in model.parameters())
# 	print('total model parameters size ', pytorch_total_params)
# 	print('trainable parameters')
	
# 	for name, param in model.named_parameters():
# 		if param.requires_grad:
# 			print (name + ', '+str(param.data.shape))
# 	print('-'*40)
# 	print('un-trainable parameters')
# 	for name, param in model.named_parameters():
# 		if not param.requires_grad:
# 			print (name, param.data.shape)

def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=1,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=False)
	argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')

	argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--aggre', type=str, default='mean')

	argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--num-batch', type=int, default=12)

	argparser.add_argument('--re-partition-method', type=str, default='REG')
	# argparser.add_argument('--re-partition-method', type=str, default='random')
	argparser.add_argument('--num-re-partition', type=int, default=0)

	# argparser.add_argument('--balanced_init_ratio', type=float, default=0.2)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=1)

	argparser.add_argument('--num-hidden', type=int, default=128)

	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='10,25')
	
	argparser.add_argument('--log-indent', type=float, default=0)
#--------------------------------------------------------------------------------------
	

	argparser.add_argument('--lr', type=float, default=1e-2)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4,
						help="Weight for L2 loss")
	argparser.add_argument("--eval", action='store_true', 
						help='If not set, we will only do the training part.')

	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	

	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)
	
	args = argparser.parse_args()
	if args.setseed:
		set_seed(args)
	device = "cpu"
	if args.GPUmem:
		see_memory_usage("-----------------------------------------before load data ")
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='pubmed':
		g, n_classes = load_pubmed()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset == 'ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		device = "cuda:0"

	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
	else:
		raise Exception('unknown dataset')
		
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	main()

