#!/bin/bash
export CUDA_VISIBLE_DEVICES=""



echo "run pytorch ..."

for tensor_type in "TensorTrainMatrix" 
do 
    export name="${tensor_type}_train_then_compress_or.log"

	dlrm_pt_bin="python tensorized_dlrm_pytorch.py"

	$dlrm_pt_bin  --tensor-type=${tensor_type} --learning-rate=0.0001 --optimizer='SGD' --load-model="saved_models/${tensor_type}" --memory-map --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-256-128" --arch-mlp-top="512-256-1" --data-generation=dataset--data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function='bce' --round-targets=True  --test-num-workers=16 --memory-map --mini-batch-size=512 --nepochs=1 --test-freq=100 --print-freq=10 --kl-multiplier=0.0  --no-kl-steps=25000 > logs/${name}.log
done
#echo "run caffe2 ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
#$dlrm_c2_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time $dlrm_extra_option 2>&1 | tee run_kaggle_c2.log

echo "done"
