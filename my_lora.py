"""
LoRA (Low-Rank Adaptation) 模块实现
用于对预训练模型进行参数高效的微调。
"""
import torch
from torch import nn


class LoRA(nn.Module):
    """
    LoRA 低秩适配器模块
    
    通过两个低秩矩阵的乘积来近似权重更新:
    W' = W + BA，其中 B ∈ R^(out_features × rank)，A ∈ R^(rank × in_features)
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 1.0):
        """
        初始化 LoRA 模块
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            rank: LoRA 的秩（rank），控制低秩矩阵的大小
            alpha: 缩放因子，用于控制 LoRA 的影响力
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 低秩矩阵 A：高斯初始化
        self.A = nn.Linear(in_features, rank, bias=False)
        # 低秩矩阵 B：零初始化（确保训练开始时 LoRA 输出为 0）
        self.B = nn.Linear(rank, out_features, bias=False)
        
        # 初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            LoRA 的输出（缩放后）
        """
        return self.B(self.A(x)) * self.scaling


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 1.0, target_modules: list = None):
    """
    对模型应用 LoRA
    
    默认只对方形线性层（如 attention 的 q, k, v, o 投影）应用 LoRA，
    可以通过 target_modules 指定目标模块名称。
    
    Args:
        model: 目标模型
        rank: LoRA 的秩
        alpha: 缩放因子
        target_modules: 目标模块名称列表，如 ['q_proj', 'v_proj']
    """
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        # 默认只对方形线性层应用 LoRA
        if isinstance(module, nn.Linear):
            should_apply = False
            
            if target_modules is not None:
                # 检查模块名称是否匹配
                for target in target_modules:
                    if target in name:
                        should_apply = True
                        break
            else:
                # 默认只对方形权重应用（通常是 attention 层）
                if module.weight.shape[0] == module.weight.shape[1]:
                    should_apply = True
            
            if should_apply:
                lora = LoRA(
                    in_features=module.weight.shape[1],
                    out_features=module.weight.shape[0],
                    rank=rank,
                    alpha=alpha
                ).to(device)
                
                setattr(module, "lora", lora)
                original_forward = module.forward
                
                # 使用闭包工厂函数来正确捕获变量
                def make_forward_with_lora(orig_forward, lora_module):
                    def forward_with_lora(x):
                        return orig_forward(x) + lora_module(x)
                    return forward_with_lora
                
                module.forward = make_forward_with_lora(original_forward, lora)


def load_lora(model: nn.Module, path: str):
    """
    从文件加载 LoRA 权重
    
    Args:
        model: 应用了 LoRA 的模型
        path: LoRA 权重文件路径
    """
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {
                k.replace(f'{name}.lora.', ''): v 
                for k, v in state_dict.items() 
                if f'{name}.lora.' in k
            }
            if lora_state:
                module.lora.load_state_dict(lora_state)


def save_lora(model: nn.Module, path: str):
    """
    保存 LoRA 权重到文件
    
    Args:
        model: 应用了 LoRA 的模型
        path: 保存路径
    """
    state_dict = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {
                f'{name}.lora.{k}': v 
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    
    torch.save(state_dict, path)


def get_lora_params(model: nn.Module) -> list:
    """
    获取模型中所有 LoRA 参数
    
    Args:
        model: 应用了 LoRA 的模型
        
    Returns:
        LoRA 参数列表
    """
    lora_params = []
    
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_params.append(param)
    
    return lora_params


def count_lora_params(model: nn.Module) -> dict:
    """
    统计 LoRA 参数量
    
    Args:
        model: 应用了 LoRA 的模型
        
    Returns:
        包含参数统计信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "lora_params": lora_params,
        "trainable_params": trainable_params,
        "lora_ratio": lora_params / total_params if total_params > 0 else 0
    }


def freeze_non_lora_params(model: nn.Module):
    """
    冻结非 LoRA 参数
    
    Args:
        model: 应用了 LoRA 的模型
    """
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False


def merge_lora(model: nn.Module):
    """
    将 LoRA 权重合并到原始权重中
    
    注意：合并后无法再单独保存/加载 LoRA 权重
    
    Args:
        model: 应用了 LoRA 的模型
    """
    for name, module in model.named_modules():
        if hasattr(module, 'lora') and isinstance(module, nn.Linear):
            lora = module.lora
            # 计算合并后的权重: W' = W + scaling * B @ A
            delta_weight = lora.scaling * lora.B.weight @ lora.A.weight
            module.weight.data += delta_weight
            
            # 移除 LoRA
            delattr(module, 'lora')
            # 恢复原始 forward
            module.forward = nn.Linear.forward.__get__(module, nn.Linear)
