#!/bin/bash
for ((i=1200; i<=1700; i=i+10))
do
python3 test.py --dataset cifar10 --exp cifar10-bCR-lambda10-1700 --epoch_id $i --num_channels 4 \
--num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --nz 50 --z_emb_dim 256 \
--n_mlp 4 --ch_mult 1 2 2 --image_size 32 --current_resolution 16 --attn_resolutions 32 \
--scale_factor 105.0 --AutoEncoder_config autoencoder/config/kl-f2.yaml --AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
--batch_size 500 --compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy
done
