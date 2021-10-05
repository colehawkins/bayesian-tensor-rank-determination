#!/bin/bash
TENSOR_TYPE='Tucker'
./bench/tensorized_dlrm.sh --save-model="saved_models/${TENSOR_TYPE}" --tensor-type=$TENSOR_TYPE --optimizer='Adam' --use-gpu=0 --test-freq=248 --print-freq=124 --learning-rate=0.005
