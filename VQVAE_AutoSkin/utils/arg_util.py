import json
import math
import os
import os.path as osp
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch

try:
    from tap import Tap
except ImportError as e:
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    time.sleep(5)
    raise e

import dist

class Args(Tap):
    exp_name: str       
    bed: str            
    resume: str = ''            
    lpips_path: str = ''        
    dino_path: str = ''         
    val_img_pattern: str = ''
    data: str = '' 
    datasets_str: str = ''
    
    zero: int = 0               
    compile_vae: int = 0        
    compile_disc: int = 0       
    compile_lpips: int = 0      
    
    ddp_static: bool = False    
    
    vae_grad_ckpt: bool = False  
    disc_grad_ckpt: bool = False 
    grad_accu: int = 1      
    prof: bool = False      
    profall: bool = False   
    tos_profiler_file_prefix: str = ''
    
    
    vae: str = 'cnn'        
    drop_path: float = 0.1  
    
    ch: int = 160
    drop_out: float = 0.05
    
    vocab_size: int = 4096
    vocab_width: int = 32
    vocab_norm: bool = False
    vq_beta: float = 0.25           
    
    
    dino_depth: int = 12        
    dino_kernel_size: int = 9   
    disc_norm: str = 'sbn'      
    disc_spec_norm: bool = True 
    disc_aug_prob: float = 1.0  
    disc_start_ep: float = 600    
    disc_warmup_ep: float = 0   
    reg: float = 0.0    
    reg_every: int = 4  
    
    
    vae_init: float = -0.5  
    vocab_init: float = -1  
    disc_init: float = 0.02 
    
    
    fp16: bool = False
    bf16: bool = False
    vae_lr: float = 3e-4    
    disc_lr: float = 3e-4   
    vae_wd: float = 0.005   
    disc_wd: float = 0.0005 
    grad_clip: float = 10   
    ema: float = 0.9999     
    
    warmup_ep: float = 0    
    wp0: float = 0.005      
    sche: str = 'cos'       
    sche_end: float = 0.3   
    
    ep: int = 600               
    lbs: int = 0                
    bs: int = 4               
    
    opt: str = 'adamw'      
    oeps: float = 0
    fuse_opt: bool = torch.cuda.is_available()      
    vae_opt_beta: str = '0.5_0.9'   
    disc_opt_beta: str = '0.5_0.9'  
    
    
    l1: float = 0.2     
    l2: float = 1.0     
    lp: float = 0.5     
    lpr: int = 48       
    ld: float = 0.4     
    le: float = 0.0     
    gada: int = 1       
    bcr: float = 4.     
    bcr_cut: float = 0.2
    dcrit: str = 'hg'   
    
    
    
    flash_attn: bool = True       
    
    
    subset: float = 1.0         
    img_size: int = 256
    mid_reso: float = 1.125     
    hflip: bool = False         
    workers: int = 8            
    
    
    local_debug: bool = 'KEVIN_LOCAL' in os.environ
    dbg_unused: bool = False
    dbg_nan: bool = False   
    
    
    cmd: str = ' '.join(sys.argv[1:])  
    
    
    
    
    
    acc_all: float = None   
    acc_real: float = None  
    acc_fake: float = None  
    last_Lnll: float = None 
    last_L1: float = None   
    last_Ld: float = None   
    last_wei_g: float = None
    grad_boom: str = None   
    diff: float = None      
    diffs: str = ''         
    diffs_ema: str = None   
    cur_phase: str = ''         
    cur_ep: str = ''            
    cur_it: str = ''            
    remain_time: str = ''       
    finish_time: str = ''       
    
    iter_speed: float = None    
    img_per_day: float = None   
    max_nvidia_smi: float = 0            
    max_memory_allocated: float = None   
    max_memory_reserved: float = None    
    num_alloc_retries: int = None        
    
    
    
    local_out_dir_path: str = os.path.join('', 'local_output')  
    tb_log_dir_path: str = ''  
    tb_log_dir_online: str = ''
    log_txt_path: str = ''           
    last_ckpt_pth_bnas: str = '...'     
    
    tf32: bool = True       
    device: str = 'cpu'     
    seed: int = None        
    deterministic: bool = False
    same_seed_for_all_ranks: int = 0     
    def seed_everything(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if self.seed is not None:
            print(f'[in seed_everything] {self.deterministic=}', flush=True)
            if self.deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            seed = self.seed + dist.get_rank()*16384
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:   
        if self.seed is None: return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g
    
    def compile_model(self, m, fast):
        if fast == 0 or self.local_debug or not hasattr(torch, 'compile'):
            return m
        mode = {
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]
        print(f'[TORCH.COMPILE: {mode=}] compile {type(m)} ...', end='', flush=True)
        stt = time.perf_counter()
        m = torch.compile(m, mode=mode)
        print(f'     finished! ({time.perf_counter()-stt:.2f}s)', flush=True, clean=True)
        return m
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        
        for k in self.class_variables.keys():
            if k not in {'device'}:     
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):  
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    def load_state_dict_vae_only(self, d: Union[OrderedDict, dict, str]):
        for k in d.keys():
            if k not in {
                'vae',  
            }:
                continue
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)
    if args.local_debug:
        args.bed = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bed')
        args.lpips_path = 
        args.dino_path = 
        args.val_img_pattern = 
        args.seed, args.deterministic = 1, True
        args.vae_init = args.disc_init = -0.5
        
        args.img_size = 64
        args.vae = 'cnn'
        args.ch = 32
        args.vocab_width = 16
        args.vocab_size = 4096
        args.disc_norm = 'gn'
        args.dino_depth = 3
        args.dino_kernel_size = 1
        args.vae_opt_beta = args.disc_opt_beta = '0.5_0.9'
        args.l2, args.l1, args.ll, args.le = 1.0, 0.2, 0.0, 0.1
    
    
    if len(args.extra_args) > 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')
    
    
    from utils import misc
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    dist.init_distributed_mode(local_out_path=args.local_out_dir_path, timeout_minutes=30)
    
    
    args.set_tf32(args.tf32)
    args.seed_everything()
    args.device = dist.get_device()
    
    if not torch.cuda.is_available() or (not args.bf16 and not args.fp16):
        args.flash_attn = False
    
    
    assert args.bed
    if args.exp_name not in args.bed:
        args.bed = osp.join(args.bed, f'{args.exp_name}')
    args.bed = args.bed.rstrip(osp.sep)
    os.makedirs(args.bed, exist_ok=True)
    
    if not args.lpips_path:
        args.lpips_path = 
    if not args.dino_path:
        args.dino_path =
    if not args.val_img_pattern:
        args.val_img_pattern = 
    if not args.tos_profiler_file_prefix.endswith('/'):
        args.tos_profiler_file_prefix += '/'
    
    
    if args.lbs == 0:
        args.lbs = max(1, round(args.bs / args.grad_accu / dist.get_world_size()))
    args.bs = args.lbs * dist.get_world_size()
    args.workers = min(args.workers, args.lbs)
    
    
    
    
    if args.warmup_ep == 0:
        args.warmup_ep = args.ep * 0.01
    if args.disc_start_ep == 0:
        args.disc_start_ep = args.ep * 0.2
    if args.disc_warmup_ep == 0:
        args.disc_warmup_ep = args.ep * 0.02
    
    
    args.log_txt_path = os.path.join(args.local_out_dir_path, 'log.txt')
    args.last_ckpt_pth_bnas = os.path.join(args.bed, f'ckpt-last.pth')
    
    _reg_valid_name = re.compile(r'[^\w\-+,.]')
    tb_name = _reg_valid_name.sub(
        '_',
        f'tb-{args.exp_name}'
        f'__{args.vae}'
        f'__b{args.bs}ep{args.ep}{args.opt[:4]}vlr{args.vae_lr:g}wd{args.vae_wd:g}dlr{args.disc_lr:g}wd{args.disc_wd:g}'
    )
    
    if dist.is_master():
        os.system(f'rm -rf {os.path.join(args.bed, "ready-node*")} {os.path.join(args.local_out_dir_path, "ready-node*")}')
    
    args.tb_log_dir_path = os.path.join(args.local_out_dir_path, tb_name)
    
    return args
