#!/bin/bash
for tensor_type in "CP" 
do for kl_mult in 0.01 0.001 0.0001
do for no_kl_steps in 100000;
do for lr in 0.005;
do
	export CUDA_VISIBLE_DEVICES=1
	name="${tensor_type}_warmup_${no_kl_steps}_${optimizer}_lr_${lr}_kl_${kl_mult}"
	./bench/tensorized_dlrm.sh  --optimizer="Adam" --learning-rate=${lr} --save-model="saved_models/${name}" --tensor-type="${tensor_type}" --use-gpu=1 --test-freq=10240 --print-freq=1024 --kl-multiplier=${kl_mult} --no-kl-steps=${no_kl_steps} > logs/${name}.log
done
done
done
done
