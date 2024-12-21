#!/bin/bash

new_model_name=Mistral-7B-Instruct-v0.3_bf16_simpo_v1
model_id=mistralai/Mistral-7B-Instruct-v0.3

pip install trl -U deepspeed
echo "${PWD}"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${PWD}/accelerate_configs/deepspeed_zero3.yaml ${PWD}/simpo.py --config ${PWD}/training_configs/simpo_v1.yaml

pip install peft -q

python3 merge_lora.py \
    --model_path=${model_id} \
    --adapter_path=models_raw/${new_model_name} \
    --new_model_path=models/${new_model_name} || echo "merge failed"