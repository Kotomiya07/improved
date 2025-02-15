#!/bin/bash
for ((i=100; i<=700; i=i+25))
do
python3 test.py --dataset lsun --exp kl-f8-256 --epoch_id $i --num_channels 4 \
--num_channels_dae 128 --num_timesteps 4 --num_res_blocks 3 --nz 50 --z_emb_dim 256 \
 --ch_mult 1 2 2 2 --image_size 256 --current_resolution 32 --attn_resolutions 16 \
--scale_factor 60.0 --AutoEncoder_config autoencoder/config/kl-f8.yaml --AutoEncoder_ckpt autoencoder/weight/kl-f8.ckpt \
--batch_size 25 --compute_fid --real_img_dir pytorch_fid/lsun_church_stat.npy
done
