import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
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
import random
from models_fix import DiT_models

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class ProbLoss(nn.Module):
    def __init__(self, opt):
        assert opt["loss_type"] in ["bce", "hinge"]
        super().__init__()
        self.loss_type = opt["loss_type"]
        self.device = opt["device"]
        self.ones = torch.ones(opt["batch_size"]).to(opt["device"])
        self.zeros = torch.zeros(opt["batch_size"]).to(opt["device"])
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, logits, condition):
        assert condition in ["gen", "disc_real", "disc_fake"]
        batch_len = len(logits)

        if self.loss_type == "bce":
            if condition in ["gen", "disc_real"]:
                return self.bce(logits, self.ones[:batch_len])
            else:
                return self.bce(logits, self.zeros[:batch_len])
        
        elif self.loss_type == "hinge":
        # SPADEでのHinge lossを参考に実装
        # https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py        
            if condition == "gen":
                # Generatorでは、本物になるようにHinge lossを返す
                return -torch.mean(logits)
            elif condition == "disc_real":
                minval = torch.min(logits - 1, self.zeros[:batch_len])
                return -torch.mean(minval)
            else:
                minval = torch.min(-logits - 1, self.zeros[:batch_len])
                return -torch.mean(minval)

class RandomCutout(object):
    def __init__(self, size=8):
        self.size = size

    def __call__(self, img_batch):
        # バッチサイズと画像の高さ、幅を取得
        batch_size, channels, height, width = img_batch.size()

        for i in range(batch_size):
            # ランダムな位置を決定
            x = random.randint(0, width - self.size)
            y = random.randint(0, height - self.size)

            # カットアウトを適用
            img_batch[i, :, y:y + self.size, x:x + self.size] = 0

        return img_batch

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

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


def train(rank, gpu, args):
    from EMA import EMA
    from score_sde.models.discriminator import Discriminator_large, Discriminator_small
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp, WaveletNCSNpp

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    #nz = args.nz  # latent dimension
    nz = args.nz = batch_size  # latent dimension

    dataset = create_dataset(args)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
    #                                                                 num_replicas=args.world_size,
    #                                                                 rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                            #   sampler=train_sampler,
                                              drop_last=True)
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    disc_net = [Discriminator_small, Discriminator_large]
    netG = DiT_models[args.model](
        input_size=args.image_size,
        in_channels=args.num_channels,
    ).to(device)
    
    if args.dataset in ['cifar10', 'stl10']:
        netD = disc_net[0](nc=2 * args.num_channels, ngf=args.ngf,
                           t_emb_dim=args.t_emb_dim,
                           act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)
    else:
        netD = disc_net[1](nc=2 * args.num_channels, ngf=args.ngf,
                           t_emb_dim=args.t_emb_dim,
                           act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)

    # broadcast_params(netG.parameters())
    # broadcast_params(netD.parameters())

    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters(
    )), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters(
    )), lr=args.lr_g, betas=(args.beta1, args.beta2))

    #if args.use_ema:
    #    optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    
    ema = deepcopy(netG).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerD, args.num_epoch, eta_min=1e-5)

    # ddp
    # netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    # netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    """############### DELETE TO AVOID ERROR ###############"""
    # Wavelet Pooling
    #if not args.use_pytorch_wavelet:
    #    dwt = DWT_2D("haar")
    #    iwt = IDWT_2D("haar")
    #else:
    #    dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
    #    iwt = DWTInverse(mode='zero', wave='haar').cuda()
        
    
    #load encoder and decoder
    config_path = args.AutoEncoder_config 
    ckpt_path = args.AutoEncoder_ckpt 
    
    #if args.dataset in ['cifar10', 'stl10'] or True:

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    AutoEncoder = instantiate_from_config(config['model'])


    checkpoint = torch.load(ckpt_path, map_location=device)
    AutoEncoder.load_state_dict(checkpoint['state_dict'])
    AutoEncoder.eval()
    AutoEncoder.to(device)
    
    #else:
    #    AutoEncoder = load_model_from_config(config_path, ckpt_path)
    """############### END DELETING ###############"""
    
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

    # 拡張の候補を定義
    augment = transforms.RandomChoice([
        transforms.RandomCrop(args.image_size, padding=4, padding_mode="reflect"), # ランダムな位置で切り取り
        transforms.RandomResizedCrop(size=args.image_size, scale=(0.8, 1.2)),      # ズームインとズームアウト
        transforms.RandomRotation(degrees=30),                                     # 左右30度までのローテート
        transforms.RandomHorizontalFlip(),                                         # ランダムフリップ
        RandomCutout(size=8),                                                      # 8x8のランダムな位置でのカットアウト
        RandomCutout(size=16),                                                     # 16x16のランダムな位置でのカットアウト
    ])
    
    print(f"AutoEncoder Parameters: {sum(p.numel() for p in AutoEncoder.parameters()):,}")
    print(f"Generator Parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in netD.parameters()):,}")

    loss = ProbLoss({"device":device, "batch_size":args.batch_size, "loss_type":args.loss_type})
    loss_mse = nn.MSELoss()
    
    update_ema(ema, netG, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start_epoch = torch.cuda.Event(enable_timing=True)
    end_epoch = torch.cuda.Event(enable_timing=True)


    for epoch in range(init_epoch, args.num_epoch + 1):
        # train_sampler.set_epoch(epoch)
        start_epoch.record()
        for iteration, (x, y) in enumerate(data_loader):
            start.record()
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            #requires_grad(netD)
            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()

            #requires_grad(netG, flag=False)
            for p in netG.parameters():
                p.requires_grad = False

            # sample from p(x_0)
            x0 = x.to(device, non_blocking=True)

            """################# Change here: Encoder #################"""
            with torch.no_grad():
                posterior = AutoEncoder.encode(x0)
                real_data = posterior.sample().detach()
            #print("MIN:{}, MAX:{}".format(real_data.min(), real_data.max()))
            real_data = real_data / args.scale_factor #300.0  # [-1, 1]
            
            
            #assert -1 <= real_data.min() < 0
            #assert 0 < real_data.max() <= 1
            """################# End change: Encoder #################"""
            # sample t
            t = torch.randint(0, args.num_timesteps,
                              (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            #print(f"{D_real[:5]=}")
            errD_real = F.softplus(-D_real).mean()
            #errD_real = loss(D_real, "disc_real")
            
            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_penalty_call(args, D_real, x_t)
            else:
                if global_step % args.lazy_reg == 0:
                    grad_penalty_call(args, D_real, x_t)

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            #print(f"{output[:5]=}")
            errD_fake = F.softplus(output).mean()
            #errD_fake = loss(output, "disc_fake")
            
            errD_fake.backward()

            torch.nn.utils.clip_grad_norm_(netD.parameters(), 0.5)

            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # update G
            #requires_grad(netD, flag=False)

            #requires_grad(netG)
            for p in netD.parameters():
                p.requires_grad = False

            for p in netG.parameters():
                p.requires_grad = True
            netG.zero_grad()

            t = torch.randint(0, args.num_timesteps,
                              (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errG = F.softplus(-output).mean()
            #errG = loss(output, "gen")

            # reconstructior loss
            if args.sigmoid_learning and args.rec_loss:
                ######alpha
                rec_loss = F.l1_loss(x_0_predict, real_data)
                errG = errG + alpha[epoch]*rec_loss

            elif args.rec_loss and not args.sigmoid_learning:
                rec_loss = F.l1_loss(x_0_predict, real_data)
                errG = errG + rec_loss
            
            
            #optimizerG.zero_grad()
            errG.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 0.5)
            optimizerG.step()

            end.record()
            torch.cuda.synchronize()

            global_step += 1
            if rank == 0:
                iter_time = start.elapsed_time(end)
                if args.sigmoid_learning:
                    print('\r[#{:05}][#{:04}][{:04.0f}ms] G Loss[{:.4f}] D Loss[{:.4f}] alpha[{:.4f}]'.format(
                        epoch, iteration, iter_time, errG.item(), errD.item(), alpha[epoch]), end="")
                elif args.rec_loss:
                    print('\r[#{:05}][#{:04}][{:04.0f}ms] G Loss[{:.4f}] D Loss[{:.4f}] rec_loss[{:.4f}]'.format(
                        epoch, iteration, iter_time, errG.item(), errD.item(), rec_loss.item()), end="")
                else:   
                    print('\r[#{:05}][#{:04}][{:04.0f}ms] G Loss[{:.4f}] D Loss[{:.4f}]'.format(
                        epoch, iteration, iter_time, errG.item(), errD.item()), end="")

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
            x_t_1 = torch.randn_like(posterior.sample())
            fake_sample = sample_from_model(
                pos_coeff, netG, args.num_timesteps, x_t_1, T, args)

            """############## CHANGE HERE: DECODER ##############"""
            fake_sample *= args.scale_factor #300
            real_data *= args.scale_factor #300
            with torch.no_grad():
                fake_sample = AutoEncoder.decode(fake_sample)
                real_data = AutoEncoder.decode(real_data)
            
            fake_sample = (torch.clamp(fake_sample, -1, 1) + 1) / 2  # 0-1
            real_data = (torch.clamp(real_data, -1, 1) + 1) / 2  # 0-1 
            
            """############## END HERE: DECODER ##############"""

            torchvision.utils.save_image(fake_sample, os.path.join(
                exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)))
            torchvision.utils.save_image(
                real_data, os.path.join(exp_path, 'real_data.png'))

            wandb.log({"fake_sample": [wandb.Image(fake_sample[:4])], "real_data": [wandb.Image(real_data[:4])]})
            

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict(),
                               'ema': ema.state_dict()}
                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                #if args.use_ema:
                #    optimizerG.swap_parameters_with_ema(
                #        store_params_in_ema=True)
                if args.use_ema:
                    update_ema(ema, netG, decay=args.ema_decay)

                torch.save(netG.state_dict(), os.path.join(
                    exp_path, 'netG_{}.pth'.format(epoch)))
                #if args.use_ema:
                #    optimizerG.swap_parameters_with_ema(
                #        store_params_in_ema=True)
                if args.use_ema:
                    update_ema(ema, netG, decay=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=12,
                        help='channel of wavelet subbands')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--patch_size', type=int, default=1,
                        help='Patchify image into non-overlapped patches')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), nargs='+', type=int,
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # generator and training
    parser.add_argument(
        '--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./data')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int,
                        default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float,
                        default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float,
                        default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float,
                        default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    # wavelet GAN
    parser.add_argument("--current_resolution", type=int, default=256)
    parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--rec_loss", action="store_true")
    parser.add_argument("--net_type", default="normal")
    parser.add_argument("--num_disc_layers", default=6, type=int)
    parser.add_argument("--no_use_fbn", action="store_true")
    parser.add_argument("--no_use_freq", action="store_true")
    parser.add_argument("--no_use_residual", action="store_true")

    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=50,
                        help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int,
                        default=25, help='save ckpt every x epochs')

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6002',
                        help='port for master')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num_workers')
    
    ##### My parameter #####
    parser.add_argument('--scale_factor', type=float, default=16.,
                        help='scale of Encoder output')
    parser.add_argument("--img_rec_loss", action="store_true", default=True)
    parser.add_argument(
        '--AutoEncoder_config', default='./autoencoder/config/autoencoder_kl_f2_16x16x4_Cifar10_big.yaml', help='path of config file for AntoEncoder')

    parser.add_argument(
        '--AutoEncoder_ckpt', default='./autoencoder/weight/last_big.ckpt', help='path of weight for AntoEncoder')
    parser.add_argument("--sigmoid_learning", action="store_true")

    
    parser.add_argument("--class_conditional", action="store_true", default=False)
    
    parser.add_argument('--loss_type', type=str, default='bce', help='loss type bce or hinge')

    # bCRのハイパーパラメータを追加
    parser.add_argument('--lambda_fake', type=float, default=2.0, help='bCR regularization strength for fake images')
    parser.add_argument('--lambda_real', type=float, default=2.0, help='bCR regularization strength for real images')
    
    # DiTのハイパーパラメータを追加
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    args = parser.parse_args()

    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' %
                  (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(
                global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')
        wandb.init(
            project="IDDGAN_DiT",
            name=args.exp,
            # すべてのパラメータをログに記録
            config={
                "seed": args.seed,
                "image_size": args.image_size,
                "num_channels": args.num_channels,
                "centered": args.centered,
                "use_geometric": args.use_geometric,
                "beta_min": args.beta_min,
                "beta_max": args.beta_max,
                "patch_size": args.patch_size,
                "num_channels_dae": args.num_channels_dae,
                "n_mlp": args.n_mlp,
                "ch_mult": args.ch_mult,
                "num_res_blocks": args.num_res_blocks,
                "attn_resolutions": args.attn_resolutions,
                "dropout": args.dropout,
                "resamp_with_conv": args.resamp_with_conv,
                "conditional": args.conditional,
                "fir": args.fir,
                "fir_kernel": args.fir_kernel,
                "skip_rescale": args.skip_rescale,
                "resblock_type": args.resblock_type,
                "progressive": args.progressive,
                "progressive_input": args.progressive_input,
                "progressive_combine": args.progressive_combine,
                "embedding_type": args.embedding_type,
                "fourier_scale": args.fourier_scale,
                "not_use_tanh": args.not_use_tanh,
                "exp": args.exp,
                "dataset": args.dataset,
                "datadir": args.datadir,
                "nz": args.nz,
                "num_timesteps": args.num_timesteps,
                "z_emb_dim": args.z_emb_dim,
                "t_emb_dim": args.t_emb_dim,
                "batch_size": args.batch_size,
                "num_epoch": args.num_epoch,
                "ngf": args.ngf,
                "lr_g": args.lr_g,
                "lr_d": args.lr_d,
                "beta1": args.beta1,
                "beta2": args.beta2,
                "no_lr_decay": args.no_lr_decay,
                "use_ema": args.use_ema,
                "ema_decay": args.ema_decay,
                "r1_gamma": args.r1_gamma,
                "lazy_reg": args.lazy_reg,
                "current_resolution": args.current_resolution,
                "use_pytorch_wavelet": args.use_pytorch_wavelet,
                "rec_loss": args.rec_loss,
                "net_type": args.net_type,
                "num_disc_layers": args.num_disc_layers,
                "no_use_fbn": args.no_use_fbn,
                "no_use_freq": args.no_use_freq,
                "no_use_residual": args.no_use_residual,
                "save_content": args.save_content,
                "save_content_every": args.save_content_every,
                "save_ckpt_every": args.save_ckpt_every,
                "num_proc_node": args.num_proc_node,
                "num_process_per_node": args.num_process_per_node,
                "node_rank": args.node_rank,
                "local_rank": args.local_rank,
                "master_address": args.master_address,
                "master_port": args.master_port,
                "num_workers": args.num_workers,
                "scale_factor": args.scale_factor,
                "img_rec_loss": args.img_rec_loss,
                "AutoEncoder_config": args.AutoEncoder_config,
                "AutoEncoder_ckpt": args.AutoEncoder_ckpt,
                "sigmoid_learning": args.sigmoid_learning,
                "class_conditional": args.class_conditional,
                "loss_type": args.loss_type,
                "lambda_fake": args.lambda_fake,
                "lambda_real": args.lambda_real,
                "model": args.model
            }
        )
        #init_processes(0, size, train, args)
        train(0, 0, args)
        wandb.finish()