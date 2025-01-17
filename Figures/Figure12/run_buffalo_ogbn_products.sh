#!/bin/bash


# save_path=./log/2-layer-10,25/SAGE/h_1024
mkdir ./log
mkdir ./log/products
save_path=./log/products/buffalo
mkdir $save_path
dataset=ogbn-products
hidden=128
epoch=4
layer=2
fanout='10,25'

for nb in  16 24 32
do
    echo "---start  $nb batches. It will cost about 2 mins."
    python buffalo_block_gen.py \
        --dataset $dataset \
        --selection-method products_25_backpack_bucketing \
        --num-batch $nb \
        --mem-constraint 18.1 \
        --num-layers $layer \
        --fan-out $fanout \
        --model SAGE \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch $epoch \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-3 \
    > ${save_path}/nb_${nb}.log
done

