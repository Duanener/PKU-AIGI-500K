import argparse
import math
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_msssim import ms_ssim

import clip
from model.cm_gru import CM_GRU
from model.dataset import MyDataset

from torch.utils.tensorboard import SummaryWriter   
import os
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch_sub(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, type='mse', is_MOD=False
):
    model.train()
    device = next(model.parameters()).device

    for i, (d, text) in enumerate(train_dataloader):
        text = clip.tokenize(text, truncate=True).to(device)
        text = Clip.encode_text(text, ).type(torch.float32)

        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d, text)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 3000 == 0:
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )


def test_epoch(epoch, test_list, model, criterion, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for loader in test_list:
                for d, text in loader:
                    text = clip.tokenize(text, truncate=True).to(device)
                    text = Clip.encode_text(text).type(torch.float32)

                    d = d.to(device)
                    out_net = model(d, text)
                    out_criterion = criterion(out_net, d)

                    aux_loss.update(model.aux_loss())
                    bpp_loss.update(out_criterion["bpp_loss"])
                    loss.update(out_criterion["loss"])
                    out_criterion["mse_loss"] = -10 * torch.log10(out_criterion["mse_loss"])
                    mse_loss.update(out_criterion["mse_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for loader in test_list:
                for d in loader:
                    d = d.to(device)
                    out_net = model(d)
                    out_criterion = criterion(out_net, d)

                    aux_loss.update(model.aux_loss())
                    bpp_loss.update(out_criterion["bpp_loss"])
                    loss.update(out_criterion["loss"])
                    ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    return loss.avg


def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + f"/{epoch}.pth.tar")
    torch.save(state, save_path + "/checkpoint_latest.pth.tar")
    # if is_best:
    #     torch.save(state, save_path + "/checkpoint_best.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Training model"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=5e-4,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=None, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N_conv", type=int, default=128,
    )
    parser.add_argument(
        "--N_trans", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    # parser.add_argument("--continue_train", action="store_true")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, f'{str(args.N_conv)}_{str(args.lmbda)}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "_tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "_tensorboard/")

    SD21B_transforms_train = transforms.Compose([transforms.RandomCrop((256, 256)), transforms.ToTensor()])
    SD21B_transforms_test = transforms.Compose([transforms.CenterCrop((512, 512)), transforms.ToTensor()])
    SD21_transforms_train = transforms.Compose([transforms.RandomCrop((256, 256)), transforms.ToTensor()])
    SD21_transforms_test = transforms.Compose([transforms.CenterCrop((768, 768)), transforms.ToTensor()])
    SDXL_transforms_train = transforms.Compose([transforms.RandomCrop((256, 256)), transforms.ToTensor()])
    SDXL_transforms_test = transforms.Compose([transforms.CenterCrop((1024, 1024)), transforms.ToTensor()])
    MJ_transforms_train = transforms.Compose([transforms.RandomCrop((256, 256)), transforms.ToTensor()])
    MJ_transforms_test = transforms.Compose([transforms.CenterCrop((1024, 1024)), transforms.ToTensor()])
    MOD_transforms_train = transforms.Compose([transforms.RandomCrop((256, 256)), transforms.ToTensor()])
    MOD_transforms_test = transforms.Compose([transforms.CenterCrop((1408, 640)), transforms.ToTensor()])

    train_SD21B = MyDataset(os.path.join(args.dataset, 'SD-2_1-B/train'), os.path.join(args.dataset, 'SD-2_1-B/train.txt'), SD21B_transforms_train)
    test_SD21B = MyDataset(os.path.join(args.dataset, 'SD-2_1-B/val'), os.path.join(args.dataset, 'SD-2_1-B/vaild.txt'), SD21B_transforms_test)
    train_SD21 = MyDataset(os.path.join(args.dataset, 'SD-2_1/train'), os.path.join(args.dataset, 'SD-2_1/train.txt'), SD21_transforms_train)
    test_SD21 = MyDataset(os.path.join(args.dataset, 'SD-2_1/val'), os.path.join(args.dataset, 'SD-2_1/vaild.txt'), SD21_transforms_test)
    train_SDXL = MyDataset(os.path.join(args.dataset, 'SD-XL/train'), os.path.join(args.dataset, 'SD-XL/train.txt'), SDXL_transforms_train)
    test_SDXL = MyDataset(os.path.join(args.dataset, 'SD-XL/val'), os.path.join(args.dataset, 'SD-XL/vaild.txt'), SDXL_transforms_test)
    train_MJ = MyDataset(os.path.join(args.dataset, 'MJ/train'), os.path.join(args.dataset, 'MJ/train.txt'), MJ_transforms_train)
    test_MJ = MyDataset(os.path.join(args.dataset, 'MJ/val'), os.path.join(args.dataset, 'MJ/vaild.txt'), MJ_transforms_test)
    train_MOD = MyDataset(os.path.join(args.dataset, 'MOD/train'), os.path.join(args.dataset, 'MOD/train.txt'), MOD_transforms_train)
    test_MOD = MyDataset(os.path.join(args.dataset, 'MOD/val'), os.path.join(args.dataset, 'MOD/vaild.txt'), MOD_transforms_test)


    train_dataloader_SD21B = DataLoader(train_SD21B, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(device == "cuda"),)
    test_dataloader_SD21B = DataLoader(test_SD21B, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"),)
    train_dataloader_SD21 = DataLoader(train_SD21, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(device == "cuda"),)
    test_dataloader_SD21 = DataLoader(test_SD21, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"),)
    train_dataloader_SDXL = DataLoader(train_SDXL, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(device == "cuda"),)
    test_dataloader_SDXL = DataLoader(test_SDXL, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"),)
    train_dataloader_MJ = DataLoader(train_MJ, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True, pin_memory=(device == "cuda"),)
    test_dataloader_MJ = DataLoader(test_MJ, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"),)
    train_dataloader_MOD = DataLoader(train_MOD, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True, pin_memory=(device == "cuda"),)
    test_dataloader_MOD = DataLoader(test_MOD, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"),)

    if args.model == 'CM_GRU':
        net = CM_GRU(num_slices=5)
    else:
        raise ValueError('No such model!')
    
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        # if args.continue_train:
        #     last_epoch = checkpoint["epoch"] + 1
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        #     aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    print('load ok!')

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        # SD21B、SD21、SDXL one epoch、MJ 4 epochs、MOD 7 epochs
        train_one_epoch_sub(net, criterion, train_dataloader_MOD, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_SD21B, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)   
        train_one_epoch_sub(net, criterion, train_dataloader_MOD, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_MJ, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_MOD, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_SD21, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_MOD, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_MJ, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_MOD, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_SDXL, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_MOD, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_MJ, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_MOD, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)
        train_one_epoch_sub(net, criterion, train_dataloader_MJ, optimizer, aux_optimizer, epoch,args.clip_max_norm, type)

        loss = test_epoch(epoch, [test_dataloader_SD21B, test_dataloader_SD21, test_dataloader_SDXL,  test_dataloader_MJ, test_dataloader_MOD], net, criterion, type)
        writer.add_scalar('test_loss', loss, epoch)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )


if __name__ == "__main__":
    device = 'cuda'
    print(device)
    Clip, _ = clip.load("ViT-B/32", device=device)
    Clip.eval()

    main(sys.argv[1:])