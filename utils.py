import os
import shutil

import torch
import torch.distributed as dist
import torchvision
from PIL import Image

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(
        backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()

def save_image(fake_data, real_data, AutoEncoder, exp_path, epoch, nrow):
    with torch.no_grad():
        fake_data = AutoEncoder.decode(fake_data)
        real_data = AutoEncoder.decode(real_data)
    MEAN = [[0.5, 0.5, 0.5],[0.3053, 0.2815, 0.2438]]
    STD  = [[0.5, 0.5, 0.5],[0.1789, 0.1567, 0.1587]]
    
    def denormalize(tensor, mean, std):
        # meanとstdをtensorの形に合わせて調整
        mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()
        std = torch.tensor(std).view(1, 3, 1, 1).cuda()
        # データを逆正規化
        tensor = tensor * std + mean
        return tensor
    
    # バッチを逆正規化
    denorm_batch_fake = denormalize(fake_data, MEAN[0], STD[0])
    denorm_batch_real = denormalize(real_data, MEAN[0], STD[0])
    # torchvision.utils.make_gridでバッチをグリッド画像に変換
    grid_img_fake = torchvision.utils.make_grid(denorm_batch_fake)
    grid_img_real = torchvision.utils.make_grid(denorm_batch_real)
    
    # TensorをPIL画像に変換して保存
    ndarr_fake = grid_img_fake.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    img_fake = Image.fromarray(ndarr_fake)
    img_fake.save(os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)))

    ndarr_real = grid_img_real.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    img_real = Image.fromarray(ndarr_real)
    img_real.save(os.path.join(exp_path, 'real_sample.png'.format(epoch)))