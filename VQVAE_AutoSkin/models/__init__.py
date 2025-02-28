from typing import Tuple

import torch.nn as nn
import torch 
from utils.arg_util import Args
from .quant import VectorQuantizer2
from .vqvae import VQVAE
from .dino import DinoDisc
from .basic_vae import Encoder


def build_vae_disc(args: Args) -> Tuple[VQVAE, DinoDisc]:
    
    for clz in (
        nn.Linear, nn.Embedding,
        nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    ):
        setattr(clz, 'reset_parameters', lambda self: None)
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10,13, 16)   
    
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=False, share_quant_resi=4, v_patch_nums=patch_nums).to(args.device)
    try:
        vae.load_state_dict(torch.load(""))
        print("ok")
    except Exception as e:  
        print(f"导入模型失败: {e}")
    disc = DinoDisc(
        device=args.device, dino_ckpt_path=args.dino_path, depth=args.dino_depth, key_depths=(2, 5, 8, 11),
        ks=args.dino_kernel_size, norm_type=args.disc_norm, using_spec_norm=args.disc_spec_norm, norm_eps=1e-6,
    ).to(args.device)

    init_weights(disc, args.disc_init)
    
    
    return vae, disc


def init_weights(model, conv_std_or_gain):
    print(f'[init_weights] {type(model).__name__} with {"std" if conv_std_or_gain > 0 else "gain"}={abs(conv_std_or_gain):g}')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            if conv_std_or_gain > 0:
                nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
            else:
                nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.bias is not None: nn.init.constant_(m.bias.data, 0.)
            if m.weight is not None: nn.init.constant_(m.weight.data, 1.)
