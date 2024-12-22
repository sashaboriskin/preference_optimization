pip install -r requierements.txt -q 

pip uninstall flash-attn -y
pip install flash_attn==2.5.8 --force-reinstall -q

pip install transformers==4.45.0 -q
pip install huggingface-hub -U
pip uninstall torchvision -y

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .

pip install bitsandbytes
pip uninstall flash-attn