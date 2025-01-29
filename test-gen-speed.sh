#!/bin/bash
# H100 Inference time: 78.60+/-5.64ms
# H100 batch 1 Inference time: 35.99+/-3.22ms
# H100 batch 100 Inference time: 65.88+/-5.66ms
# A100 Inference time: 135.21+/-3.46ms
# A100 batch 1 Inference time: 58.23+/-17.20ms
# A100 batch 100 Inference time: 98.17+/-12.47ms
#python3 test_dit_no_ddp.py --dataset cifar10 --exp cifar10-ori-dit-no-ddp-s2-skip-connection-predict-noise-plus2 --epoch_id 1700 --num_channels 4 \
#--num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --nz 50 --z_emb_dim 256 \
#--n_mlp 4 --ch_mult 1 2 2 --image_size 32 --current_resolution 16 --attn_resolutions 32 \
#--scale_factor 105.0 --AutoEncoder_config autoencoder/config/kl-f2.yaml --AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
#--batch_size 100 --measure_time --real_img_dir pytorch_fid/cifar10_train_stat.npy --model DiT-S/2

# python3 test_dit.py --dataset cifar10 --exp cifar10-ori-dit-ss1-fix --epoch_id 1500 --num_channels 4 \
# --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --nz 100 --z_emb_dim 256 \
# --n_mlp 4 --ch_mult 1 2 2 --image_size 32 --current_resolution 16 --attn_resolutions 32 \
# --scale_factor 105.0 --AutoEncoder_config autoencoder/config/kl-f2.yaml --AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
# --batch_size 100 --measure_time --real_img_dir pytorch_fid/cifar10_train_stat.npy --model DiT-SS/1


# H100 batch 1 Inference time: 60.98+/-0.95ms
# H100 batch 100 Inference time: 96.50+/-2.30ms
# A100 batch 1 Inference time: 90.32+/-8.63ms
# A100 batch 100 Inference time: 144.86+/-17.42ms
# RTX4090 batch 1 Inference time: 25.08+/-2.44ms 
# RTX4090 batch 100 Inference time: 63.38+/-0.78ms
# python3 test.py --dataset cifar10 --exp kl-f2-3 --epoch_id 1825 --num_channels 4 \
# --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --nz 50 --z_emb_dim 256 \
# --n_mlp 4 --ch_mult 1 2 2 --image_size 32 --current_resolution 16 --attn_resolutions 32 \
# --scale_factor 105.0 --AutoEncoder_config autoencoder/config/kl-f2.yaml --AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
# --batch_size 100 --measure_time --real_img_dir pytorch_fid/cifar10_train_stat.npy


# H100 batch 1 Inference time: 58.56+/-3.41ms
# H100-01 batch 25 Inference time: 237.74+/-66.62ms 
# H100 batch 25 Inference time: 292.73+/-2.91ms
# H100 batch 100 Inference time: 1035.70+/-5.33ms
# A100 batch 1 Inference time: 86.44+/-11.92ms
# A100 batch 25 Inference time: 414.55+/-2.83ms
# A100 batch 100 Inference time: 1452.89+/-3.14ms
# RTX4090 batch 1 Inference time: 32.34+/-2.69ms 
# RTX4090 batch 25 Inference time: 520.90+/-1.12ms 
# RTX4090 batch 100 
# python3 test.py --dataset celeba_256 --image_size 256 --exp vq-f4-256 --epoch_id 500 --num_channels 3 \
# --num_channels_dae 128 --num_timesteps 2 --num_res_blocks 2 --nz 100 --z_emb_dim 256 \
# --n_mlp 3 --ch_mult 1 2 2 2 --image_size 256 --current_resolution 64 --attn_resolutions 16 \
# --scale_factor 6.0 --AutoEncoder_config autoencoder/config/vq-f4.yaml --AutoEncoder_ckpt autoencoder/weight/vq-f4.ckpt \
# --batch_size 25 --measure_time --real_img_dir pytorch_fid/celebahq_stat.npy

# RTX4090 batch 25 Inference time: 351.59+/-151.17ms
python3 test.py --dataset lsun --image_size 256 --exp vq-f8-256-6 --epoch_id 475 --num_channels 4 \
--num_channels_dae 128 --num_timesteps 2 --num_res_blocks 2 --nz 100 --z_emb_dim 256 \
--n_mlp 3 --ch_mult 1 2 2 2 --image_size 256 --current_resolution 32 --attn_resolutions 16 \
--scale_factor 6.0 --AutoEncoder_config autoencoder/config/vq-f8.yaml --AutoEncoder_ckpt autoencoder/weight/vq-f8.ckpt \
--batch_size 25 --measure_time --real_img_dir pytorch_fid/lsun_church_stat.npy