
<!-- ![Description of the image](main_res_OOM.png)   -->

## For different aggregator  
- Mini batch train  

To get the result of Figure 2 (a)   
`./run_mini_batch_train_diff_aggregator.sh`   
OGBN-products  
You can find the max CUDA memory consumption    
Mean ggreagator:  5.44GB the last line in log/mini_batch_train/aggregators/2_layer_aggre_mean.log   
Pool ggreagator: 8.9GB the last line in log/mini_batch_train/aggregators/2_layer_aggre_pool.log  
Lstm aggregator: OOM  


- Micro batch train (Buffalo) 

To get the result of Figure 13 (a)   
Micro batch train can finish GraphSAGE model + lstm aggregator train without OOM.   
`./run_micro_batch_train_diff_aggregator.sh`  
When the full batch data split into 12 micro batch, it can break the memory wall.  
The max cuda memory consumption is about 21.2 GB.   
more detail you can find in `log/micro_batch_train/2_layer_aggre_lstm_batch_12.log`


## For different layers  

 
- Figure 2 (b)  
Run `./run_mini_batch_train_diff_layers.sh`.  
The result of mini batch train with different layers are located in `log/mini_batch_train/layers/`  
You can find the cuda memory consumption of different model layers (Graph SAGE model +lstm aggreagtor, OGBN-arxiv)
1 layer: 4.32GB  
2 layer: 13.64GB  
3 layer: OOM   

- Figure 13 (b)  
When GPU meomry constraint is 24 GB.   
To break the memory wall, Buffalo use 2 micro batches to train 3-layer model.
`./run_micro_batch_train_diff_layers.sh` it might takes a few minutes.
Then you will get the CUDA memory consumption of 3-layer model in file `log/micro_batch_train/3_layer_aggre_lstm_batch_2.log`, which is 22.07GB. 


To save time, we only provide these two example to denote Our Buffalo breaks the memory capacity constraint in Figure 2.







