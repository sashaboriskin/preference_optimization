#!/bin/bash

pip install -r requierements.txt -q 

pip uninstall flash-attn -y
pip install flash_attn==2.5.8 --force-reinstall -q

pip install transformers==4.45.0 -q
pip install huggingface-hub==0.27.0
pip uninstall torchvision -y

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .

pip uninstall bitsandbytes -y
pip install bitsandbytes==0.45.0
pip uninstall flash-attn

pip install trl==0.13.0
pio install deepspeed==0.16.2
pip install peft==0.10.0

pip uninstall flash-attn
pip install flash_attn==2.7.2