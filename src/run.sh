export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false



for bs in 32;
do
    for dataset in mnli_test_1k;
    do
        for llm in google/flan-t5-base google/flan-t5-large google/flan-t5-xl;
        do
            python run_llm.py $llm ../datasets/$dataset.csv --out_dir ../temp_outs/ --max_gen_tokens 100 --bs $bs
        done
    done
done

