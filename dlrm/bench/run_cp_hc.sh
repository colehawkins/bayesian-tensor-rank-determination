#!/bin/bash
export tensor_type="CP"
export no_kl_steps=50000
export minibatch_size=512
export prior_type="half_cauchy"
export lr=0.005

export CUDA_VISIBLE_DEVICES=0
for kl_mult in 0.5 0.1;
do for eta in 1.0 0.1 0.01; 
do
	name="${tensor_type}_warmup_${no_kl_steps}_${optimizer}_lr_${lr}_kl_${kl_mult}_batch${minibatch_size}_eta_${eta}"
  
	dlrm_pt_bin="python tensorized_dlrm_pytorch.py"


	$dlrm_pt_bin  --nepochs=2 \
			--prior-type=$prior_type \
			--eta=$eta \
			--arch-sparse-feature-size=128 \
			--arch-mlp-bot="13-512-256-256-128" \
			--arch-mlp-top="512-256-1" \
			--data-generation=dataset \
			--data-set=kaggle \
			--raw-data-file=./input/train.txt \
			--processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
			--loss-function=bce \
			--round-targets=True \
			--test-num-workers=16 \
			--memory-map \
			--mini-batch-size=$minibatch_size  \
			--optimizer="Adam" \
			--learning-rate=${lr} \
			--tensor-type="${tensor_type}" \
			--use-gpu=1 \
			--test-freq=10240 \
			--print-freq=1024 \
			--kl-multiplier=${kl_mult} \
			--no-kl-steps=${no_kl_steps} > logs/${name}.log
done
done
echo "done"

#--save-model="saved_models/${name}" \
#--test-freq=1024 --print-freq=1024 --mini-batch-size=128
