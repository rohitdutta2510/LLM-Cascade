export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

for bs in 4;
do
    for dataset in cnn_dm_train_1k cnn_dm_test_1k;
    do
        # for llm in microsoft/Phi-3-mini-128k-instruct microsoft/Phi-3-small-128k-instruct microsoft/Phi-3-medium-128k-instruct;
        for llm in microsoft/Phi-3-small-128k-instruct;
        do
            python run_llm.py $llm ../datasets/$dataset.csv --out_dir ../temp_outs/ --max_gen_tokens 100 --bs $bs
        done
    done
done

