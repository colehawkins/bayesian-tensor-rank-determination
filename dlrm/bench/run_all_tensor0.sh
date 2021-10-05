#!/bin/bash
for tensor_type in  "TensorTrainMatrix" "Tucker"
do for optimizer in "Adam"
do for no_kl_steps in 100000;
do for lr in 0.001 0.005;
do
        export CUDA_VISIBLE_DEVICES=0
	name="${tensor_type}_warmup_${no_kl_steps}_${optimizer}_${lr}"
	./bench/tensorized_dlrm.sh  --optimizer=${optimizer} --learning-rate=${lr} --save-model="saved_models/${name}" --tensor-type="${tensor_type}" --use-gpu=1 --test-freq=10240 --print-freq=1024 --kl-multiplier=1.0 --no-kl-steps=${no_kl_steps} > logs/${name}.log
done
done
done
done
