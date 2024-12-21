CUDA_VISIBLE_DEVICES=0 python run_src/do_discriminate.py \
    --model_ckpt microsoft/Phi-3-mini-4k-instruct \
    --root_dir /zhome/ff/8/213294/rStar-main/run_outputs/GSM8K/Mistral-7B-v0.1/2024-11-12_13-32-50---[debug] \
    --dataset_name GSM8K \
    --note default
