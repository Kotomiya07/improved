#!/bin/sh
export MASTER_PORT=6038
echo MASTER_PORT=${MASTER_PORT}

export PYTHONPATH=$(pwd):$PYTHONPATH

CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

DATASET=$1
MODE=$2
GPUS=$3

if [ -z "$1" ]; then
   GPUS=1
fi

echo $DATASET $MODE $GPUS

# ----------------- IDDGAN -----------
if [[ $MODE == train ]]; then
	echo "==> Training IDDGAN"

	if [[ $DATASET == cifar10 ]]; then
		python3 train_iddgan_lab.py --dataset cifar10 --exp cifar10-san --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning

	elif [[ $DATASET == cifar10_cond ]]; then
		python3 train_iddgan_lab.py --dataset cifar10 --exp atn32_g122_2block_d3_Recloss_SmL_256_cond --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--save_content_every 1 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--class_conditional
	
	elif [[ $DATASET == cifar10_feature ]]; then
		python3 train_iddgan_feature.py --dataset cifar10 --exp cifar10-feature-kl-f2 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/kl-f2.ckpt \
			--no_lr_decay \
			--rec_loss \
			--sigmoid_learning

	elif [[ $DATASET == cifar10_cond_feature ]]; then
		python3 train_iddgan_feature.py --dataset cifar10 --exp cifar10-cond-feature-kl-f2 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/kl-f2.ckpt \
			--no_lr_decay \
			--rec_loss \
			--sigmoid_learning \
			--class_conditional

	elif [[ $DATASET == coco_256 ]]; then
		python3 train_iddgan_lab.py --dataset coco --image_size 256 --exp g1222_128_2block_d4_attn16_nz50_256 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 8 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/coco \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 256 --attn_resolution 16 --num_disc_layers 4 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/COCO_config.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/kl-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning
	
	elif [[ $DATASET == coco_128 ]]; then
		python3 train_iddgan_lab.py --dataset coco --image_size 128 --exp coco_128 --num_channels 4 --num_channels_dae 128 --ch_mult 1 2 2 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/coco \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 32 --num_disc_layers 3 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/kl-f2.ckpt \
			--scale_factor 60.0 \
			--no_lr_decay \
			--sigmoid_learning
	
	elif [[ $DATASET == coco_128_feature ]]; then
		python3 train_iddgan_feature.py --dataset coco --image_size 128 --exp coco-128-feature --num_channels 4 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 50 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/coco \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 32 --num_disc_layers 3 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/kl-f2.ckpt \
			--scale_factor 60.0 \
			--no_lr_decay \
			--sigmoid_learning
	
	elif [[ $DATASET == coco_64 ]]; then
		python3 train_iddgan_lab_celeba.py --dataset coco --image_size 64 --exp g1222_128_2block_d4_attn16_nz50_64 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 200 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 50 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/coco \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 32 --num_disc_layers 4 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/vq-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning 

	elif [[ $DATASET == celeba_256 ]]; then
		python3 train_iddgan_celeba.py --dataset celeba_256 --image_size 256 --exp g1222_128_2block_d4_attn16_2step_SmL_500ep --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 4 --rec_loss \
			--save_content_every 5 \
			--AutoEncoder_config ./autoencoder/config/CELEBA_config.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/CELEBA_weight.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning 

	elif [[ $DATASET == lsun ]]; then
		python3 train_iddgan.py --dataset lsun --image_size 256 --exp g12222_128_2block_d4_attn16_nz50_tanh --num_channels 4 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 3 --batch_size 8 --num_epoch 1000 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
			--nz 50 --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/lsun/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 32 --attn_resolution 16 --num_disc_layers 4  \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/LSUN_config.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/LSUN_weight.ckpt \
			--scale_factor 60.0 \
			--sigmoid_learning \
			--no_lr_decay 
	fi

else
	echo "==> Testing IDDGAN"
	if [[ $DATASET == cifar10 ]]; then
		python3 test_iddgan.py --dataset cifar10 --exp atn32_g122_2block_d3_Recloss_nz50_SmL --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --nz 50 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 --epoch_id 950 \
			--image_size 32 --current_resolution 16 --attn_resolutions 32 \
			--scale_factor 105.0 \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--batch_size 256 \
			--compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy 

	elif [[ $DATASET == cifar10_cond ]]; then
		python3 test_iddgan.py --dataset cifar10 --exp cifar-10-cond --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 --epoch_id 2000 \
			--image_size 32 --current_resolution 16 --attn_resolutions 32 \
			--scale_factor 105.0 \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--batch_size 256 \
			--compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy \
			--class_conditional

	elif [[ $DATASET == celeba_256 ]]; then
		python3 test_iddgan_celeba.py --dataset celeba_256 --image_size 256 --exp g1222_128_2block_d4_attn16_2step_SmL --num_channels 3 --num_channels_dae 128 \
			--nz 100 --z_emb_dim 256  --ch_mult 1 2 2 2  --num_timesteps 2 --num_res_blocks 2  --epoch_id 725 \
			--current_resolution 64 --attn_resolutions 16 \
			--AutoEncoder_config ./autoencoder/config/CELEBA_config.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/CELEBA_weight.ckpt \
			--scale_factor 6.0 \
			--batch_size 32 \
			--compute_fid --real_img_dir pytorch_fid/celebahq_stat.npy 

	elif [[ $DATASET == lsun ]]; then
		python3 test_iddgan.py --dataset lsun --image_size 256 --exp g1222_128_3block_d4_attn16_PixelRecloss_2000ep --num_channels 4 --num_channels_dae 128 \
			--ch_mult 1 2 2 2  --num_timesteps 4 --num_res_blocks 3  --epoch_id 625 \
			--current_resolution 32 --attn_resolutions 16 \
			--compute_fid --compute_fid --real_img_dir pytorch_fid/lsun_church_stat.npy \
			--AutoEncoder_config ./autoencoder/config/LSUN_config.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/LSUN_weight.ckpt \
			--scale_factor 60.0 \
			--batch_size 48 
	fi
fi