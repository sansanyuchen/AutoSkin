
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  
from models import VQVAE, build_vae_var

MODEL_DEPTH = 16    
assert MODEL_DEPTH in {16, 20, 24, 30}



hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'

vae_ckpt,var_ckpt = ,




patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')


vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    
        device=device, patch_nums=patch_nums,
        num_classes=7, depth=MODEL_DEPTH, shared_aln=False,
)


print("ok")

var_ckpt =  torch.load(var_ckpt)

vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)


var.load_state_dict(var_ckpt['trainer']['var_wo_ddp'],  strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')


import os
import torch
import random
import numpy as np
from PIL import Image as PImage


num_sampling_steps = 250   
cfg = 4                    
more_smooth = False        
batch_size = 64          
tf32 = True                


class_counts = {
    0: 327,  
    1: 514,  
    2: 1099,  
    3: 115,  
    4: 1113,  
    5: 6705,  
    6: 142   
}


class_to_folder = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}


torch.backends.cudnn.allow_tf32 = tf32
torch.backends.cuda.matmul.allow_tf32 = tf32
torch.set_float32_matmul_precision('high' if tf32 else 'highest')
device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')


save_dir = 
os.makedirs(save_dir, exist_ok=True)


for class_label, total_count in class_counts.items():
    if total_count <= 0:
        continue
    
    
    folder_name = class_to_folder.get(class_label, f"class_{class_label}")
    class_dir = os.path.join(save_dir, folder_name)
    os.makedirs(class_dir, exist_ok=True)
    
    
    num_batches = (total_count + batch_size - 1) // batch_size
    generated_count = 0
    
    for batch_idx in range(num_batches):
        current_batch = min(batch_size, total_count - generated_count)
        
        
        seed = random.randint(0, 2**32-1)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16):
                
                label_tensor = torch.full((current_batch,), class_label, 
                                       device=device, dtype=torch.long)
                
                
                images = var.autoregressive_infer_cfg(
                    B=current_batch,
                    label_B=label_tensor,
                    cfg=cfg,
                    top_k=900,
                    top_p=0.95,
                    g_seed=seed,
                    more_smooth=more_smooth
                )

        
        for img_idx in range(current_batch):
            
            img_data = images[img_idx].permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
            img = PImage.fromarray(img_data)
            
            
            global_idx = generated_count + img_idx
            filename = f"{folder_name}_seed{seed}_num{global_idx:04d}.png"
            save_path = os.path.join(class_dir, filename)
            
            
            img.save(save_path)
            print(f"finish: {save_path}")
        
        generated_count += current_batch

print("ok")
