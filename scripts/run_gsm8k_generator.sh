export PYTHONPATH=$PYTHONPATH:/zhome/ff/8/213294/rStar-main

CUDA_VISIBLE_DEVICES=1 python /zhome/ff/8/213294/rStar-main/run_src/do_generate.py \
    --dataset_name GSM8K \
    --test_json_filename test_0_99 \
    --model_ckpt deepseek-ai/deepseek-math-7b-instruct \
    --note debug \
    --half_precision \
    --num_rollouts 16

