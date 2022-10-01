# 2CET-GAN
2 cycles expression transfer GAN (2CET-GAN) is a Pixel to Pixel Human Facial Expression Transfer GAN.
This is a pytorch implementation.

Train:
```bash
python main.py \
--mode train \
--resume_iter 0 \
--eval_model_step 0 \
--test False \
--to_grey False \
--img_size 128 \
--code_dim 32 \
--encoder_grey True \
--lr_e 0.0001 \
--lambda_reg 0.5 \
--lambda_cyc_n 1.0 \
--lambda_cyc_e_config 0,100000,1.0,1.0 \
--lambda_c_e_config 5000,30000,1.0,1.0 \
--lambda_ds_e_config 0,100000,1.0,0.0 \
--train_dir ${DATASET_ROOT}/train \
--test_dir ${DATASET_ROOT}/test \
--eval_dir /content/eval \
--batch_size 8 \
--output_dir ${DATA_ROOT}/output \
--models_dir ${DATA_ROOT}/models \
--output_every 500 \
--save_every 5000 \
--total_iter 100000

```
