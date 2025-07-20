import torch
from .unet import UNet
from .lunext import LUNeXt

def create_unet(in_channels=3, out_channels=1):
    """
    創建 UNet 模型的輔助函數
    
    Args:
        in_channels: 輸入通道數
        out_channels: 輸出通道數
        
    Returns:
        UNet 模型實例
    """
    model = UNet(in_channels, out_channels)
    return model

def create_lunext(in_channels=3, out_channels=1):
    """
    創建 LUNeXt 模型的輔助函數
    
    Args:
        in_channels: 輸入通道數
        out_channels: 輸出通道數
        
    Returns:
        LUNeXt 模型實例
    """
    model = LUNeXt(in_channels, out_channels)
    return model

def save_model(model, filepath):
    """
    保存訓練好的模型
    
    Args:
        model: 訓練好的模型
        filepath: 保存路徑
    """
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存至: {filepath}")
