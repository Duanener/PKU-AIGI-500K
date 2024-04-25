import torch
import torch.nn.functional as F
from torchvision import transforms
from model.cm_gru import CM_GRU
import torch
import os
import sys
import math
import argparse
import time
import clip
from pytorch_msssim import ms_ssim
from PIL import Image


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--data_i", type=str, help="Path to image")
    parser.add_argument("--data_t", type=str, help="Path to text")
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    p = 128
    path = args.data_i
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)

    text_list = open(args.data_t, 'r').readlines()
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'

    Clip, _ = clip.load("ViT-B/32", device=device)
    Clip.eval()

    if args.model == 'CM_GRU':
        net = CM_GRU(num_slices=5)
    else:
        raise ValueError('No such Model!')
    net = net.to(device)
    net.eval()
    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    dictory = {}
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
    if args.real:
        net.update()
        print(img_list)
        img_list.sort()
        for img_name in img_list:
            print(img_name)
            
            first = img_name.split('.')[0]
            first = int(first)
            text = text_list[first-1][:-1]
            t = clip.tokenize(text, truncate=True).to(device)
            t = Clip.encode_text(t).type(torch.float32)

            img_path = os.path.join(path, img_name)
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
            x = img.unsqueeze(0)
            x_padded, padding = pad(x, p)
            count += 1
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_enc = net.compress(x_padded, t)
                out_dec = net.decompress(out_enc["strings"], t, out_enc["shape"])
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)

                img_0 = transforms.ToPILImage()(out_dec['x_hat'][0].squeeze())
                img_0.save('./tmp.png')
                out_dec["x_hat"] = transforms.ToTensor()(Image.open('./tmp.png').convert('RGB')).to(device).unsqueeze(0)
                os.remove('./tmp.png')

                num_pixels = x.size(0) * x.size(2) * x.size(3)
                
                with open('./tmp.bin', 'wb') as f:
                    for i in out_enc["strings"]:
                        f.write(bytes(i[0]))
                    f.write(text.encode())
                Size = os.path.getsize('./tmp.bin')
                os.remove('./tmp.bin')
                print(f'Bitrate: %.2fbpp' % (Size * 8.0 / num_pixels))
                print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
                print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
                Bit_rate += Size * 8.0 / num_pixels
                PSNR += compute_psnr(x, out_dec["x_hat"])
                MS_SSIM += compute_msssim(x, out_dec["x_hat"])
    else:
        raise ValueError("You shouldn't use the bitrate by entropy model when testing")

    PSNR = PSNR / count
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    total_time = total_time / count
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    print(f'average_time: {total_time:.3f} ms')


if __name__ == "__main__":

    main(sys.argv[1:])
    
