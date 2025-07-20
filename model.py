# 為了向後兼容而保留的導入和簡單包裝函數
# 建議新代碼直接從相應的模組導入

# 從模組化文件中導入功能
from models.model_utils import create_unet, create_lunext, save_model
from trainer.training_utils import train_model
from models.losses import SelfRegLoss
from test_helper.utils import (
    dice_coefficient, 
    calculate_metrics, 
    save_results, 
    save_prediction_images,
    save_results_with_images
)
from test_helper.skeleton import (
    save_skeleton_results_with_images,
    evaluate_model_skeleton,
    skeletonize_mask,
    calculate_skeleton_metrics
)

# 向後兼容的別名 - 將來可能會被移除
# 建議新代碼直接從相應模組導入
__all__ = [
    'create_unet',
    'create_lunext', 
    'save_model',
    'train_model',
    'SelfRegLoss',
    'dice_coefficient',
    'calculate_metrics',
    'save_results',
    'save_prediction_images', 
    'save_results_with_images',
    'save_skeleton_results_with_images',
    'evaluate_model_skeleton',
    'skeletonize_mask',
    'calculate_skeleton_metrics'
]