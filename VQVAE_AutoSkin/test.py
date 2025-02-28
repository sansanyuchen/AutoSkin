import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from torch.utils.data import DataLoader
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  
from utils.data import build_dataset, pil_load
from models import build_vae_disc, VQVAE, DinoDisc
MODEL_DEPTH = 16    
assert MODEL_DEPTH in {16, 20, 24, 30}
from utils import arg_util, misc

        
if __name__ == '__main__':
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    vae_wo_ddp, disc_wo_ddp = build_vae_disc(args)


    
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    ckpt = torch.load('', map_location='cpu')

    vae_wo_ddp.load_state_dict(ckpt['trainer']['vae_wo_ddp'], strict=True)
    vae = vae_wo_ddp.to(args.device)

    for p in vae_wo_ddp.parameters(): p.requires_grad_(False)
    for p in vae_wo_ddp.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')
    dataset_train, dataset_val, val_transform = build_dataset(datasets_str=args.data, subset_ratio=args.subset, final_reso=args.img_size, mid_reso=args.mid_reso, hflip=args.hflip)
    ld_val = DataLoader(
            dataset=dataset_val,  
            batch_size=1,   
            num_workers=args.workers,  
            pin_memory=True,  
            shuffle=False,  
            drop_last=False,  
        )
    iters_val = len(ld_val)

    
    output_dir = ''
    os.makedirs(output_dir, exist_ok=True)

    
    for it, inp in enumerate(ld_val):
        inp = inp.to(args.device, non_blocking=True)
        with torch.no_grad():
            rec_B3HW, *_ = vae(inp)

        
        inp_img = inp[0].cpu().numpy()  
        rec_img = rec_B3HW[0].cpu().numpy()  

        
        inp_img = (inp_img + 1) / 2  
        rec_img = (rec_img + 1) / 2

        
        inp_img = np.transpose(inp_img, (1, 2, 0))  
        rec_img = np.transpose(rec_img, (1, 2, 0))

        
        print(f"Input image value range: {inp_img.min()} to {inp_img.max()}")
        print(f"Reconstructed image value range: {rec_img.min()} to {rec_img.max()}")

        
        inp_pil = PImage.fromarray((inp_img * 255).astype('uint8'))
        rec_pil = PImage.fromarray((rec_img * 255).astype('uint8'))

        
        inp_pil.save(f'{output_dir}/input_{it}.png')
        rec_pil.save(f'{output_dir}/reconstructed_{it}.png')

        
        print(f"Saved input_{it}.png and reconstructed_{it}.png to {output_dir}")
        
        if it >= 100:
            break
