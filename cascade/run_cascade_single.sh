export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

echo -e ">> Running boolq_single_scorer.py\n"
python boolq_single_scorer.py