import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datasets_prep.dataset import create_dataset
from diffusion import sample_from_model, sample_posterior, \
    q_sample_pairs, get_time_schedule, \
    Posterior_Coefficients, Diffusion_Coefficients
#from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
#from pytorch_wavelets import DWTForward, DWTInverse
from torch.multiprocessing import Process
from utils import init_processes, copy_source, broadcast_params
import yaml

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import wandb
from copy import deepcopy
from collections import OrderedDict
from PIL import Image as PILImage
from torchmetrics.image.fid import FrechetInceptionDistance

def load_model_from_config(config_path, ckpt):
    print(f"Loading model from {ckpt}")
    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt, map_location="cpu")
    #global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model = model.first_stage_model
    model.cuda()
    model.eval()
    del m
    del u
    del pl_sd
    return model

def grad_penalty_call(args, D_real, x_t):
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True
    )[0]
    grad_penalty = (
        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()

    grad_penalty = args.r1_gamma / 2 * grad_penalty
    grad_penalty.backward()

# %%
def train(rank, gpu, args):
    from EMA import EMA
    from score_sde.models.discriminator import Discriminator_large, Discriminator_small
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp, WaveletNCSNpp

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    dataset = create_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              sampler=train_sampler,
                                              drop_last=True)
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    disc_net = [Discriminator_small, Discriminator_large]
    print("GEN: {}, DISC: {}".format(gen_net, disc_net))
    netG = gen_net(args).to(device)
    print("model loaded!")

    if args.dataset in ['cifar10', 'stl10']:
        netD = disc_net[0](nc=2 * args.num_channels, ngf=args.ngf,
                           t_emb_dim=args.t_emb_dim,
                           act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)
    else:
        netD = disc_net[1](nc=2 * args.num_channels, ngf=args.ngf,
                           t_emb_dim=args.t_emb_dim,
                           act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)

    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters(
    )), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters(
    )), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    # ddp
    netG = nn.parallel.DistributedDataParallel(
        netG, device_ids=[gpu])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    config_path = args.AutoEncoder_config
    ckpt_path = args.AutoEncoder_ckpt

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    AutoEncoder = instantiate_from_config(config['model'])

    checkpoint = torch.load(ckpt_path, map_location=device)
    AutoEncoder.load_state_dict(checkpoint['state_dict'])
    AutoEncoder.eval()
    AutoEncoder.to(device)

    num_levels = int(np.log2(args.ori_image_size // args.current_resolution))

    exp = args.exp
    parent_dir = "./saved_info/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models',
                            os.path.join(exp_path, 'score_sde/models'))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume or os.path.exists(os.path.join(exp_path, 'content.pth')):
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        # load G
        netG.load_state_dict(checkpoint['netG_dict'])
        #optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        #optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])

        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    '''Sigmoid learning parameter'''
    gamma = 6
    beta = np.linspace(-gamma, gamma, args.num_epoch+1)
    alpha = 1 - 1 / (1+np.exp(-beta))

    print(f"AutoEncoder Parameters: {sum(p.numel() for p in AutoEncoder.parameters()):,}")
    print(f"Generator Parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in netD.parameters()):,}")

    list_augments = [
        torchvision.transforms.RandomCrop(args.image_size, padding=4, padding_mode="reflect"), # ランダムな位置で切り取り
        torchvision.transforms.RandomResizedCrop(size=args.image_size, scale=(0.8, 1.2)),    # ズームインとズームアウト
        torchvision.transforms.RandomRotation(degrees=30),                                   # 左右30度までのローテート
    ]

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start_epoch = torch.cuda.Event(enable_timing=True)
    end_epoch = torch.cuda.Event(enable_timing=True)

    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Prepare a fixed set of real images for FID calculation
    if rank == 0:
        real_images_fid = []
        num_fid_images = 100 # You can adjust the number of real images for FID
        count = 0
        with torch.no_grad():
            for x, _ in data_loader:
                real_x0 = x.to(device, non_blocking=True)
                posterior_real = AutoEncoder.encode(real_x0)
                real_data_fid_batch = posterior_real.sample().detach()
                real_data_fid_batch *= args.scale_factor
                real_data_fid_batch = AutoEncoder.decode(real_data_fid_batch)
                real_data_fid_batch = (torch.clamp(real_data_fid_batch, -1, 1) + 1) / 2
                real_images_fid.append(real_data_fid_batch)
                count += real_x0.size(0)
                if count >= num_fid_images:
                    break
        real_images_fid = torch.cat(real_images_fid[:(num_fid_images + batch_size -1) // batch_size * batch_size])[:num_fid_images] # Ensure consistent size
        # Accumulate statistics for real images once
        fid.update(real_images_fid * 255, real=True)
        del real_images_fid
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()

    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)
        start_epoch.record()
        for iteration, (x, y) in enumerate(data_loader):
            start.record()
            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()

            for p in netG.parameters():
                p.requires_grad = False

            # sample from p(x_0)
            real_x0 = x.to(device, non_blocking=True)

            """################# Change here: Encoder #################"""
            with torch.no_grad():
                posterior_real = AutoEncoder.encode(real_x0)
                real_data = posterior_real.sample().detach()
            real_data = real_data / args.scale_factor #300.0  # [-1, 1]

            """################# End change: Encoder #################"""
            # sample t
            t = torch.randint(0, args.num_timesteps,
                              (real_data.size(0),), device=device)

            real_x_t, real_x_tp1 = q_sample_pairs(coeff, real_data, t)
            real_x_t.requires_grad = True

            # Train discriminator with real data
            D_real = netD(real_x_t, t, real_x_tp1.detach()).view(-1)
            errD_real = F.softplus(-D_real).mean()

            # Apply consistency regularization for real data
            augment = torchvision.transforms.RandomChoice(list_augments)
            if args.lambda_real > 0:
                # Apply augmentation to real data (define your augmentation function t)
                augmented_real_data = augment(real_data.detach())

                augmented_real_x_t, augmented_real_x_tp1 = q_sample_pairs(coeff, augmented_real_data, t)
                D_augmented_real = netD(augmented_real_x_t, t, augmented_real_x_tp1.detach()).view(-1)
                errD_real_consistency = F.mse_loss(D_real, D_augmented_real)
                errD_real = errD_real + args.lambda_real * errD_real_consistency

            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_penalty_call(args, D_real, real_x_t)
            else:
                if global_step % args.lazy_reg == 0:
                    grad_penalty_call(args, D_real, real_x_t)

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)
            fake_x0_predict = netG(real_x_tp1.detach(), t, latent_z) # Use real_x_tp1 and t for generating fake samples

            fake_x_t, fake_x_tp1 = q_sample_pairs(coeff, fake_x0_predict, t)

            output = netD(fake_x_t, t, fake_x_tp1.detach()).view(-1)
            errD_fake = F.softplus(output).mean()

            # Apply consistency regularization for fake data (bCR)
            if args.lambda_fake > 0:
                # Apply augmentation to generated fake data (define your augmentation function T_fake)
                augmented_fake_x0_predict = augment(fake_x0_predict.detach()) # Detach to avoid gradient flow through generator

                augmented_fake_x_t, augmented_fake_x_tp1 = q_sample_pairs(coeff, augmented_fake_x0_predict, t)
                D_augmented_fake = netD(augmented_fake_x_t, t, augmented_fake_x_tp1.detach()).view(-1)
                errD_fake_consistency = F.mse_loss(output, D_augmented_fake)
                errD_fake = errD_fake + args.lambda_fake * errD_fake_consistency

            errD_fake.backward()

            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # update G
            for p in netD.parameters():
                p.requires_grad = False

            for p in netG.parameters():
                p.requires_grad = True
            netG.zero_grad()

            t_gen = torch.randint(0, args.num_timesteps,
                              (real_data.size(0),), device=device)
            x_t_gen, x_tp1_gen = q_sample_pairs(coeff, real_data, t_gen) # Use real data for generator training

            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1_gen.detach(), t_gen, latent_z)
            x_pos_predict = sample_posterior(pos_coeff, x_0_predict, x_tp1_gen, t_gen)

            output = netD(x_pos_predict, t_gen, x_tp1_gen.detach()).view(-1)
            errG = F.softplus(-output).mean()

            # reconstruction loss
            if args.sigmoid_learning and args.rec_loss:
                ######alpha
                rec_loss = F.l1_loss(x_0_predict, real_data)
                errG = errG + alpha[epoch]*rec_loss

            elif args.rec_loss and not args.sigmoid_learning:
                rec_loss = F.l1_loss(x_0_predict, real_data)
                errG = errG + rec_loss

            errG.backward()
            optimizerG.step()

            end.record()
            torch.cuda.synchronize()

            global_step += 1
            if rank == 0:
                iter_time = start.elapsed_time(end)
                log_message = f'\r[#{epoch:05}][#{iteration:04}][{iter_time:04.0f}ms] G Loss[{errG.item():.4f}] D Loss[{errD.item():.4f}]'
                if args.sigmoid_learning:
                    log_message += f' alpha[{alpha[epoch]:.4f}]'
                elif args.rec_loss:
                    log_message += f' rec_loss[{rec_loss.item():.4f}]'
                print(log_message, end="")

                wandb.log({"G_loss_iter": errG.item(), "D_loss_iter": errD.item(), "iter_time": iter_time})

        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()

        if rank == 0:
            end_epoch.record()
            torch.cuda.synchronize()
            epoch_time = start_epoch.elapsed_time(end_epoch)
            print("\nEpoch time: {:.3f} s".format(epoch_time / 1000))
            wandb.log({"G_loss": errG.item(), "D_loss": errD.item(), "alpha": alpha[epoch], "epoch_time": epoch_time / 1000})
            ########################################
            x_t_1 = torch.randn_like(posterior_real.sample())
            fake_sample = sample_from_model(
                pos_coeff, netG, args.num_timesteps, x_t_1, T, args)

            """############## CHANGE HERE: DECODER ##############"""
            fake_sample *= args.scale_factor #300
            # real_data *= args.scale_factor #300 # No need to decode real_data here, already done for FID
            with torch.no_grad():
                fake_sample = AutoEncoder.decode(fake_sample)

            fake_sample = (torch.clamp(fake_sample, -1, 1) + 1) / 2  # 0-1
            # real_data = (torch.clamp(real_data, -1, 1) + 1) / 2  # 0-1

            """############## END HERE: DECODER ##############"""

            torchvision.utils.save_image(fake_sample, os.path.join(
                exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)))
            # torchvision.utils.save_image(
            #     real_data, os.path.join(exp_path, 'real_data.png'))

            # Calculate FID
            with torch.no_grad():
                fid.update(fake_sample * 255, real=False)
                current_fid = fid.compute()
                wandb.log({"FID": current_fid})
                fid.reset() # Reset FID metric for the next iteration

            wandb.log({"fake_sample": [wandb.Image(fake_sample[:4])]}) #, "real_data": [wandb.Image(real_data[:4])]})

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('\nSaving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(
                        store_params_in_ema=True)

                torch.save(netG.state_dict(), os.path.join(
                    exp_path, 'netG_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(
                        store_params_in_ema=True)