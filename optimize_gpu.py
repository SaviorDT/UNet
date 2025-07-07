import torch
import sys
import os
import gc

def optimize_system_for_deep_learning():
    """
    優化系統設置以提高深度學習訓練性能
    """
    print("=== 系統優化工具 ===")
    
    # 檢查 PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 檢查並優化 CUDA 設置
    if torch.cuda.is_available():
        cuda_ver = torch.version.cuda
        print(f"CUDA 版本: {cuda_ver}")
        device_count = torch.cuda.device_count()
        print(f"檢測到 {device_count} 個 CUDA 設備")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_cap = torch.cuda.get_device_capability(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {device_name} (架構: {device_cap[0]}.{device_cap[1]}, 記憶體: {total_mem:.2f} GB)")
            
            # 清理 GPU 記憶體
            torch.cuda.empty_cache()
            
            # 檢查是否支援 TensorFloat-32 (TF32)
            if device_cap[0] >= 8:  # Ampere 及以上架構支援 TF32
                print(f"  - GPU {i} 支援 TensorFloat-32 (TF32)")
                torch.set_float32_matmul_precision('high')
            
        # 設定 cudnn 參數
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            print("已啟用 cuDNN 自動調優模式 (torch.backends.cudnn.benchmark = True)")
        
        # 檢查 cudnn 是否可用
        if torch.backends.cudnn.is_available():
            print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        else:
            print("警告: cuDNN 不可用，這可能會導致訓練速度減慢")
            
        # 設置環境變數優化 CUDA 行為
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print("已設置 CUDA 環境變數以優化性能")
        
        try:
            # 使用 pynvml 檢查 GPU 溫度和功率狀態
            import pynvml
            pynvml.nvmlInit()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 獲取溫度
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                print(f"  GPU {i} 溫度: {temp}°C")
                
                # 獲取風扇速度
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                    print(f"  GPU {i} 風扇速度: {fan}%")
                except:
                    pass
                
                # 檢查限頻
                try:
                    throttled = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                    if throttled > 0:
                        print(f"  警告: GPU {i} 可能處於限頻狀態")
                        if throttled & pynvml.nvmlClocksThrottleReasonHwThermalSlowdown:
                            print("    - 溫度過高導致降頻")
                        if throttled & pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown:
                            print("    - 功率限制導致降頻")
                except:
                    pass
        except:
            print("無法獲取 GPU 詳細狀態訊息，請安裝 pynvml: pip install pynvml")
    else:
        print("警告: 未檢測到 CUDA 設備，將使用 CPU 訓練 (速度會很慢)")
    
    # 檢查系統記憶體
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"系統記憶體: 總計 {vm.total/1024**3:.2f} GB, 可用 {vm.available/1024**3:.2f} GB ({vm.percent}% 已使用)")
        
        # 釋放記憶體
        gc.collect()
        
        # 檢查和優化 CPU 設定
        cpu_count = psutil.cpu_count(logical=True)
        print(f"CPU 核心數: {cpu_count} (邏輯核心)")
        
        # 檢查是否有足夠的核心用於 DataLoader
        rec_workers = min(4, cpu_count - 1)
        print(f"推薦的 DataLoader workers 數: {rec_workers}")
        
        # 檢查系統負載
        load = psutil.getloadavg()[0] / cpu_count * 100
        if load > 70:
            print(f"警告: 系統負載較高 ({load:.1f}%)，可能影響訓練效能")
    except ImportError:
        print("無法獲取系統資源訊息，請安裝 psutil: pip install psutil")
    
    print("\n=== 優化建議 ===")
    print("1. 確保 GPU 驅動是最新的")
    print("2. 監控 GPU 溫度，避免過熱導致降頻")
    print("3. 關閉其他耗費 GPU 資源的程式")
    print("4. 使用較小的批次大小 (batch size) 減少記憶體使用")
    print("5. 如果持續遇到效能問題，可能需要減少模型大小或優化資料載入流程")
    
    return True

if __name__ == "__main__":
    optimize_system_for_deep_learning()
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "check_only":
            print("\n系統檢查完畢，未啟動訓練程序")
            sys.exit(0)
            
        print("\n正在匯入訓練模組...")
        from trainer.k_fold import main as train_main
        
        print("\n開始執行訓練程序...")
        train_main()
    except KeyboardInterrupt:
        print("\n程序被使用者中斷")
    except Exception as e:
        print(f"\n執行時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理資源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
