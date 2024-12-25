#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Use args: $0 <file.py> <config.yaml>"
    exit 1
fi

po_script=$1
config_file=$2
po_base_name=$(basename "$po_script" .py)

new_model_name="Mistral-7B-Instruct-v0.3_bf16_${po_base_name}"
model_id=mistralai/Mistral-7B-Instruct-v0.3

echo "${PWD}"
echo "New model name: ${new_model_name}"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${PWD}/accelerate_configs/deepspeed_zero3.yaml ${PWD}/${po_script} --config ${PWD}/training_configs/${config_file}

python3 merge_lora.py \
    --model_path=${model_id} \
    --adapter_path=models_raw/${new_model_name} \
    --new_model_path=models/${new_model_name} || echo "merge failed"