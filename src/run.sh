export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false



for bs in 4;
do
    for dataset in boolq_test_1k;
    do
        for llm in microsoft/Phi-3-medium-4k-instruct;
        do
            python run_llm.py $llm ../datasets/$dataset.csv --out_dir ../temp_outs/ --max_gen_tokens 100 --bs $bs
        done
    done
done

