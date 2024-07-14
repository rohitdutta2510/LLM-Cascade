export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

echo -e ">> Running boolq.py\n"
python boolq.py  
