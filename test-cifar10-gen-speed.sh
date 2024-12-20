#!/bin/bash
# H100 Inference time: 78.60+/-5.64ms
# A100 Inference time: 135.21+/-3.46ms
# python3 test_dit.py --dataset cifar10 --exp cifar10-ori-dit-ss1-fix --epoch_id 1500 --num_channels 4 \
# --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --nz 100 --z_emb_dim 256 \
# --n_mlp 4 --ch_mult 1 2 2 --image_size 32 --current_resolution 16 --attn_resolutions 32 \
# --scale_factor 105.0 --AutoEncoder_config autoencoder/config/kl-f2.yaml --AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
# --batch_size 100 --measure_time --real_img_dir pytorch_fid/cifar10_train_stat.npy --model DiT-SS/1

python3 test.py --dataset cifar10 --exp kl-f2-3 --epoch_id 1825 --num_channels 4 \
--num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --nz 50 --z_emb_dim 256 \
--n_mlp 4 --ch_mult 1 2 2 --image_size 32 --current_resolution 16 --attn_resolutions 32 \
--scale_factor 105.0 --AutoEncoder_config autoencoder/config/kl-f2.yaml --AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
--batch_size 100 --measure_time --real_img_dir pytorch_fid/cifar10_train_stat.npy 