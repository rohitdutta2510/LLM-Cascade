export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

echo -e ">> Running mnli_single_scorer.py\n"
python mnli_single_scorer.py
