import dist
import torch
from typing import Tuple
from utils import arg_util


def debug_nan_grad(model):
    print('[debug_nan_grad opened]')
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        if p.requires_grad and p.grad is not None:
            if p.grad.isnan().any():
                err = f"[rk{dist.get_rank()}] [{n} (type={type(p)}, shape={tuple(p.shape)}, num={p.grad.isnan().sum().item()}/{p.numel()})] grad has NAN"
                print(err, flush=True, force=True, deeper=True)
                raise AttributeError(err)
            if p.grad.isinf().any():
                err = f"[rk{dist.get_rank()}] [{n} (type={type(p)}, shape={tuple(p.shape)}, num={p.grad.isinf().sum().item()}/{p.numel()})] grad has INF"
                print(err, flush=True, force=True, deeper=True)
                raise AttributeError(err)


def debug_nan_param(model):
    print('[debug_nan_param opened]')
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        if p.data.isnan().any():
            err = f"[rk{dist.get_rank()}] [{n} (type={type(p)}, shape={tuple(p.shape)}, num={p.isnan().sum().item()}/{p.numel()})] param has NAN"
            print(err, flush=True, force=True, deeper=True)
            raise AttributeError(err)
        if p.data.isinf().any() and 'attn_bias' not in n and 'attn_mask' not in n:
            err = f"[rk{dist.get_rank()}] [{n} (type={type(p)}, shape={tuple(p.shape)}, num={p.isinf().sum().item()}/{p.numel()})] param has INF"
            print(err, flush=True, force=True, deeper=True)
            raise AttributeError(err)


def debug_nan_hook(model):
    print('[debug_nan_hook opened]')
    
    Tensors = Tuple[torch.Tensor]
    
    def pre_f_hook(module, inps: Tensors):
        if not module.training:
            return
        if inps is not None:
            for x in inps:
                if isinstance(x, torch.Tensor):
                    d = x.data
                    if d.isnan().any():
                        err = f"[rk{dist.get_rank()}] [module={type(module)}] [==preforward==] inps has NAN (shape={tuple(d.shape)}, num={d.isnan().sum().item()}/{d.numel()})"
                        print(err, flush=True, force=True, deeper=True)
                        raise AttributeError(err)
                    if d.isinf().any():
                        err = f"[rk{dist.get_rank()}] [module={type(module)}] [==preforward==] inps has INF (shape={tuple(d.shape)}, num={d.isinf().sum().item()}/{d.numel()})"
                        print(err, flush=True, force=True, deeper=True)
                        raise AttributeError(err)
        
    
    def f_hook(module, inps: Tensors, oups: Tensors):
        if not module.training:
            return
        if oups is not None:
            for x in oups:
                if isinstance(x, torch.Tensor):
                    d = x.data
                    if d.isnan().any():
                        err = f"[rk{dist.get_rank()}] [module={type(module)}] [==forward==] oups has NAN (shape={tuple(d.shape)}, num={d.isnan().sum().item()}/{d.numel()})"
                        print(err, flush=True, force=True, deeper=True)
                        raise AttributeError(err)
                    if d.isinf().any():
                        err = f"[rk{dist.get_rank()}] [module={type(module)}] [==forward==] oups has INF (shape={tuple(d.shape)}, num={d.isinf().sum().item()}/{d.numel()})"
                        print(err, flush=True, force=True, deeper=True)
                        raise AttributeError(err)
        
    
    def b_hook(module, g_inps: Tensors, g_oups: Tensors):
        if not module.training:
            return
        if g_inps is not None:
            for x in g_inps:
                if isinstance(x, torch.Tensor):
                    d = x.data
                    if d.isnan().any():
                        err = f"[rk{dist.get_rank()}][ [module={type(module)}] ==backward==] g_inps has NAN (shape={tuple(d.shape)}, num={d.isnan().sum().item()}/{d.numel()})"
                        print(err, flush=True, force=True, deeper=True)
                        raise AttributeError(err)
                    if d.isinf().any():
                        err = f"[rk{dist.get_rank()}][ [module={type(module)}] ==backward==] g_inps has INF (shape={tuple(d.shape)}, num={d.isinf().sum().item()}/{d.numel()})"
                        print(err, flush=True, force=True, deeper=True)
                        raise AttributeError(err)
        if g_oups is not None:
            for x in g_oups:
                if isinstance(x, torch.Tensor):
                    d = x.data
                    if d.isnan().any():
                        err = f"[rk{dist.get_rank()}][ [module={type(module)}] ==backward==] g_oups has NAN (shape={tuple(d.shape)}, num={d.isnan().sum().item()}/{d.numel()})"
                        print(err, flush=True, force=True, deeper=True)
                        raise AttributeError(err)
                    if d.isinf().any():
                        err = f"[rk{dist.get_rank()}][ [module={type(module)}] ==backward==] g_oups has INF (shape={tuple(d.shape)}, num={d.isinf().sum().item()}/{d.numel()})"
                        print(err, flush=True, force=True, deeper=True)
                        raise AttributeError(err)
        
    
    for n, m in model.named_modules():
        
        if not isinstance(m, (torch.nn.Identity, torch.nn.ModuleList)):
            m.register_forward_pre_hook(pre_f_hook)
            m.register_forward_hook(f_hook)
            
            
            
            
            
