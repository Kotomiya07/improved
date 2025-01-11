#!/bin/sh
export PYTHONPATH=$(pwd):$PYTHONPATH

CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

DATASET=$1
MODE=$2
GPUS=$3
PORT=$4

if [ -z "$1" ]; then
   GPUS=1
fi

if [ -z "$4" ] ; then
	PORT=51234
fi

export MASTER_PORT=${PORT}
echo MASTER_PORT=${MASTER_PORT}

echo $DATASET $MODE $GPUS

# ----------------- IDDGAN -----------
if [[ $MODE == train ]]; then
	echo "==> Training IDDGAN"

	if [[ $DATASET == cifar10 ]]; then
		python3 train_iddgan.py --dataset cifar10 --exp cifar10-test --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning

	elif [[ $DATASET == cifar10-vgg ]]; then
		python3 train_iddgan_vgg.py --dataset cifar10 --exp cifar10-vgg --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--use_vgg_loss
	
	elif [[ $DATASET == cifar10-nz100-seed42 ]]; then
		python3 train_iddgan.py --dataset cifar10 --exp cifar10-nz100-seed42 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--seed 42

	elif [[ $DATASET == cifar10-sinarctan ]]; then
		python3 train_iddgan.py --dataset cifar10 --exp cifar10-sinarctan --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--alpha_type sinarctan

	elif [[ $DATASET == cifar10-tanh ]]; then
		python3 train_iddgan.py --dataset cifar10 --exp cifar10-tanh --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--alpha_type tanh

	elif [[ $DATASET == cifar10-no-rec ]]; then
		python3 train_iddgan.py --dataset cifar10 --exp cifar10-no-rec --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--sigmoid_learning
	
	elif [[ $DATASET == cifar10-bCR ]]; then
            python3 train_iddgan_bCR.py --dataset cifar10 --exp cifar10-bCR-lambda1.5 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--lambda_fake 1.5 \
			--lambda_real 1.5
	
	elif [[ $DATASET == cifar10-onestep ]]; then
		python3 train_iddgan_onestep.py --dataset cifar10 --exp cifar10-onestep --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning

	elif [[ $DATASET == cifar10-vq ]]; then
		python3 train_iddgan_celeba.py --dataset cifar10 --exp cifar10-vq-sf6 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 6.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/vq-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/vq-f2.ckpt \
			--rec_loss \
			--sigmoid_learning

	elif [[ $DATASET == cifar10-no-ddp ]]; then
		python3 train_iddgan_no_ddp.py --dataset cifar10 --exp cifar10-no-ddp --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning
	
	elif [[ $DATASET == cifar10-a100-01 ]]; then
		accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_iddgan.py --dataset cifar10 --exp cifar10-ddp --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port 50719 --num_process_per_node 1 --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--num_proc_node 2 --node_rank 0 --master_address 10.111.5.21
	
	elif [[ $DATASET == cifar10-a100-02 ]]; then
		torchrun --nnodes=2 --nproc-per-node=1 --node-rank=1 --rdzv-id=719 --rdzv-backend=c10d --rdzv-endpoint=10.111.5.21:50719 --master-port 50719 train_iddgan.py --dataset cifar10 --exp cifar10-ddp --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port 50719 --num_process_per_node 1 --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--num_proc_node 2 --node_rank 1 --master_address 10.111.5.21
	
	elif [[ $DATASET == cifar10-dit ]]; then
		python3 train_iddgan_dit.py --dataset cifar10 --exp cifar10-dit --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 0.8e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning
	
	elif [[ $DATASET == cifar10-dit-fix ]]; then
		python3 train_iddgan.py --dataset cifar10 --exp cifar10-dit-fix-3 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--resblock_type biggan_with_dit
	
    elif [[ $DATASET == cifar10-dit-no-ddp-ss1 ]]; then
		python3 train_iddgan_dit_no_ddp.py --dataset cifar10 --exp cifar10-ori-dit-no-ddp-ss1-hinge --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.2e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-SS/1 \
			--loss_type hinge

    elif [[ $DATASET == cifar10-dit-no-ddp-s2-bce ]]; then
		python3 train_iddgan_dit_no_ddp.py --dataset cifar10 --exp cifar10-ori-dit-no-ddp-s2-bce --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.2e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-S/2 \
			--loss_type bce

    elif [[ $DATASET == cifar10_dit_no_ddp_m2 ]]; then
		python3 train_iddgan_dit_no_ddp.py --dataset cifar10 --exp cifar10-ori-dit-no-ddp-m2 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.2e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-M/2

    elif [[ $DATASET == cifar10_dit_no_ddp_m2_hinge ]]; then
		python3 train_iddgan_dit_no_ddp.py --dataset cifar10 --exp cifar10-ori-dit-no-ddp-m2-hinge --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.2e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-M/2 \
			--loss_type hinge

    elif [[ $DATASET == cifar10_dit_ss1_fix ]]; then
		python3 train_iddgan_dit.py --dataset cifar10 --exp cifar10-ori-dit-ss1-fix --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.2e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-SS/1
	
    elif [[ $DATASET == cifar10-dit-s2 ]]; then
		python3 train_iddgan_dit.py --dataset cifar10 --exp cifar10-ori-dit-s2-skip-y --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 2000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.2e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-S/2

	elif [[ $DATASET == cifar10_dit_b2 ]]; then
		python3 train_iddgan_dit.py --dataset cifar10 --exp cifar10-ori-dit-b2 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 8e-5 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-B/2
	
	elif [[ $DATASET == cifar10_dit_l2 ]]; then
		python3 train_iddgan_dit.py --dataset cifar10 --exp cifar10-ori-dit-l2 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 8e-5 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-L/2
	
	elif [[ $DATASET == cifar10_dit_xl2 ]]; then
		python3 train_iddgan_dit.py --dataset cifar10 --exp cifar10-ori-dit-xl2 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.0e-4 --lr_g 2.0e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-XL/2
	
	elif [[ $DATASET == cifar10_dit_no_ddp_s2 ]]; then
		python3 train_iddgan_dit_no_ddp.py --dataset cifar10 --exp cifar10-ori-dit-no-ddp-s2-skip-connection-predict-noise-plus-epoch3000 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 3000 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.2e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-S/2
	
	elif [[ $DATASET == cifar10_dit_no_ddp_b2 ]]; then
		python3 train_iddgan_dit_no_ddp.py --dataset cifar10 --exp cifar10-ori-dit-no-ddp-b2 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-B/2
	
	elif [[ $DATASET == cifar10_dit_no_ddp_l2 ]]; then
		python3 train_iddgan_dit_no_ddp.py --dataset cifar10 --exp cifar10-ori-dit-no-ddp-l2 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-L/2
	
	elif [[ $DATASET == cifar10_dit_no_ddp_xl2 ]]; then
		python3 train_iddgan_dit_no_ddp.py --dataset cifar10 --exp cifar10-ori-dit-no-ddp-xl2 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--model DiT-XL/2
	
	elif [[ $DATASET == cifar10_dit_fix_acc ]]; then
		accelerate launch --mixed_precision fp16 train_iddgan.py --dataset cifar10 --exp cifar10-dit-fix-2 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
			--sigmoid_learning \
			--resblock_type biggan_with_dit
    
    elif [[ $DATASET == cifar10_bCR_hinge ]]; then
            python3 train_iddgan_bCR.py --dataset cifar10_no_transform --exp cifar10-bCR-fix-hinge --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1700 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 5 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--rec_loss \
            --loss_type hinge \
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
		python3 train_iddgan_lab.py --dataset coco --image_size 256 --exp coco-256 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 2 \
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
		python3 train_iddgan_lab_celeba.py --dataset coco --image_size 64 --exp coco-64 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
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
	
	elif [[ $DATASET == afhq-cat-256-kl-f4 ]]; then
		python3 train_iddgan_lab.py --dataset afhq_cat --image_size 256 --exp cat-256-kl-f4 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 50 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/afhq \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 32 --num_disc_layers 4 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/kl-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/kl-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning
	
	elif [[ $DATASET == afhq-cat-256-kl-f2 ]]; then
		python3 train_iddgan_lab.py --dataset afhq_cat --image_size 256 --exp cat-256-kl-f2 --num_channels 4 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 16 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 50 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/afhq \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 128 --attn_resolution 32 --num_disc_layers 4 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/kl-f2.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning
	
	elif [[ $DATASET == afhq-cat-256-kl-f2-feature ]]; then
		python3 train_iddgan_feature.py --dataset afhq_cat --image_size 256 --exp cat-256-kl-f2-feature --num_channels 4 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 16 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 50 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/afhq \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 128 --attn_resolution 32 --num_disc_layers 4 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/kl-f2.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning
	
	elif [[ $DATASET == afhq-cat-256-vq-f4 ]]; then
		python3 train_iddgan_lab.py --dataset afhq_cat --image_size 256 --exp cat-256-vq-f4 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 50 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/afhq \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 32 --attn_resolution 32 --num_disc_layers 4 --rec_loss \
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
			--AutoEncoder_config ./autoencoder/config/vq-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning
	
	elif [[ $DATASET == celeba-128 ]]; then
		python3 train_iddgan_celeba.py --dataset celeba_128 --image_size 128 --exp celeba-128-vqf4 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 32 --attn_resolution 16 --num_disc_layers 4 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/vq-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning
	
	elif [[ $DATASET == celeba-64 ]]; then
		python3 train_iddgan_celeba.py --dataset celeba_64 --image_size 64 --exp celeba-64-vqf4 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 16 --attn_resolution 16 --num_disc_layers 4 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/vq-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning
	
	elif [[ $DATASET == celeba-32 ]]; then
		python3 train_iddgan_celeba.py --dataset celeba_32 --image_size 32 --exp celeba-32-vqf4 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 8 --attn_resolution 16 --num_disc_layers 3 --rec_loss \
			--save_content_every 5 \
			--AutoEncoder_config ./autoencoder/config/vq-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning

	elif [[ $DATASET == celeba-256-dit-b2 ]]; then
		python3 train_iddgan_dit_celeba.py --dataset celeba_256 --image_size 256 --exp celeba-256-dit-b2 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 4 --rec_loss \
			--save_content_every 5 \
			--AutoEncoder_config ./autoencoder/config/vq-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning \
			--model DiT-B/2
	
	elif [[ $DATASET == celeba_256_dit_no_ddp_xl2 ]]; then
		python3 train_iddgan_dit_no_ddp.py --dataset celeba_256 --image_size 256 --exp celeba-256-dit-no-dddp-xl2 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 4 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/vq-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning \
			--model DiT-XL/2
	
	elif [[ $DATASET == celeba_256_dit_no_ddp ]]; then
		python3 train_iddgan_no_ddp_celeba.py --dataset celeba_256 --image_size 256 --exp celeba-256-dit-no-ddp --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 4 --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/vq-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning \
			--resblock_type biggan_with_dit
    
    elif [[ $DATASET == celeba-256-bCR ]]; then
		python3 train_iddgan_celeba_bCR.py --dataset celeba_256 --image_size 256 --exp vq-f4-256-bCR-lambda3 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 4 --rec_loss \
			--save_content_every 5 --save_ckpt_every 5\
			--AutoEncoder_config ./autoencoder/config/vq-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f4.ckpt \
			--scale_factor 6.0 \
			--no_lr_decay \
			--sigmoid_learning 

	elif [[ $DATASET == lsun ]]; then
		python3 train_iddgan.py --dataset lsun --image_size 256 --exp g12222_128_2block_d4_attn16_nz50_tanh --num_channels 4 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 3 --batch_size 32 --num_epoch 4000 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
			--nz 50 --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/lsun/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 32 --attn_resolution 16 --num_disc_layers 4  \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/LSUN_config.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/LSUN_weight.ckpt \
			--scale_factor 60.0 \
			--sigmoid_learning \
			--no_lr_decay 

	elif [[ $DATASET == lsun_vqf8_no_ddp ]]; then
		python3 train_iddgan_no_ddp_celeba.py --dataset lsun --image_size 256 --exp lsun_vqf8_no_ddp --num_channels 4 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 3 --batch_size 32 --num_epoch 4000 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
			--nz 50 --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/lsun/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 32 --attn_resolution 16 --num_disc_layers 4  \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/vq-f8.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f8.ckpt \
			--scale_factor 60.0 \
			--sigmoid_learning \
			--no_lr_decay 
	
	elif [[ $DATASET == lsun_vqf4_likeCelebA_dit_l2 ]]; then
		python3 train_iddgan_dit_celeba.py --dataset lsun --image_size 256 --exp lsun_vqf4_likeceleba_dit_l2 --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 4000 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/lsun/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 4  --rec_loss \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/vq-f4.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f4.ckpt \
			--scale_factor 6.0 \
			--sigmoid_learning \
			--no_lr_decay \
			--model DiT-L/2

	elif [[ $DATASET == lsun_vqf8_dit_no_ddp_l2 ]]; then
		python3 train_iddgan_dit_no_ddp_celeba.py --dataset lsun --image_size 256 --exp lsun_vqf8_dit_no_ddp_L2 --num_channels 4 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 3 --batch_size 32 --num_epoch 4000 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
			--nz 50 --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/lsun/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 32 --attn_resolution 16 --num_disc_layers 4  \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/vq-f8.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/vq-f8.ckpt \
			--scale_factor 60.0 \
			--sigmoid_learning \
			--no_lr_decay \
			--model DiT-L/2
    
    elif [[ $DATASET == lsun_bCR_kl ]]; then
		python3 train_iddgan_bCR.py --dataset lsun_no_transform --image_size 256 --exp kl-f8-256-bCR --num_channels 4 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 3 --batch_size 32 --num_epoch 4000 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
			--nz 50 --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/lsun/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 1\
			--current_resolution 32 --attn_resolution 16 --num_disc_layers 4  \
			--save_content_every 1 \
			--AutoEncoder_config ./autoencoder/config/kl-f8.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/kl-f8.ckpt \
			--scale_factor 60.0 \
			--sigmoid_learning \
			--no_lr_decay 
    
    elif [[ $DATASET == lsun_bCR_vq ]]; then
		python3 train_iddgan_celeba_bCR.py --dataset lsun_no_transform --image_size 256 --exp vq-f8-256-bCR --num_channels 4 \
        --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 3 --batch_size 8 --num_epoch 1000 --ngf 64 \
        --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
        --lr_d 1.0e-4 --lr_g 2.0e-4 --lazy_reg 10 --ch_mult 1 2 2 2 --save_content --datadir data/lsun/ \
        --master_port 6088 --num_process_per_node 1 --save_content_every 1 \
        --current_resolution 32 --attn_resolutions 16 --num_disc_layers 4 \
        --scale_factor 60.0 --no_lr_decay \
        --AutoEncoder_config autoencoder/config/vq-f8.yaml \
        --AutoEncoder_ckpt autoencoder/weight/vq-f8.ckpt \
        --rec_loss \
        --sigmoid_learning
	fi

else
	echo "==> Testing IDDGAN"
	if [[ $DATASET == cifar10 ]]; then
		python3 test_iddgan.py --dataset cifar10 --exp kl-f2-4 --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --nz 50 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 --epoch_id 1300 \
			--image_size 32 --current_resolution 16 --attn_resolutions 32 \
			--scale_factor 105.0 \
			--AutoEncoder_config autoencoder/config/kl-f2.yaml \
			--AutoEncoder_ckpt autoencoder/weight/kl-f2.ckpt \
			--batch_size 250 \
			--fid_only --real_img_dir pytorch_fid/cifar10_train_stat.npy 

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

   elif [[ $DATASET == afhq_cat ]]; then
		python3 test_iddgan.py --dataset afhq_cat --image_size 256 --exp  --num_channels 4 --num_channels_dae 128 \
			--ch_mult 1 2 2 2  --num_timesteps 4 --num_res_blocks 3  --epoch_id 625 \
			--current_resolution 32 --attn_resolutions 16 \
			--compute_fid --compute_fid --real_img_dir pytorch_fid/lsun_church_stat.npy \
			--AutoEncoder_config ./autoencoder/config/LSUN_config.yaml \
			--AutoEncoder_ckpt ./autoencoder/weight/LSUN_weight.ckpt \
			--scale_factor 60.0 \
			--batch_size 48
	fi
fi
