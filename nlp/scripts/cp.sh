#!/bin/bash

#may need smaller batch size to fit on 8gb gpu
BATCH_SIZE=256


for tensor_type in  'CP';
do for kl_mult in 0.0 5e-4 1e-4 5e-5;
do python train.py  --n_epochs=100 --embedding ${tensor_type} --rank 100 --lr 0.0005 --kl-multiplier $kl_mult --rank-loss True --batch-size $BATCH_SIZE | tee logs/${tensor_type}_low_batch_${kl_mult}.txt;
done
done



