#!/bin/bash
for ((i=300; i<=500; i=i+25))
do
python3 test_no_ddp_celeba.py --dataset celeba_256 --exp celeba-256-dit-no-ddp --epoch_id $i --num_channels 3 \
--num_channels_dae 128 --num_timesteps 2 --num_res_blocks 2 --nz 100 --z_emb_dim 256 \
--n_mlp 3 --ch_mult 1 2 2 2 --image_size 256 --current_resolution 64 --attn_resolutions 16 \
--scale_factor 6.0 --AutoEncoder_config autoencoder/config/vq-f4.yaml --AutoEncoder_ckpt autoencoder/weight/vq-f4.ckpt \
--batch_size 100 --compute_fid --real_img_dir pytorch_fid/celebahq_stat.npy --resblock_type biggan_with_dit
done