#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# > 0 ]]; then
    dlrm_extra_option=$*
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

dlrm_pt_bin="python tensorized_dlrm_pytorch.py"

echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
$dlrm_pt_bin   --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-256-128" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --print-time $dlrm_extra_option --test-num-workers=16 --memory-map   2>&1 | tee run_kaggle_pt.log

echo "done"


#--test-freq=1024 --print-freq=1024 --mini-batch-size=128
