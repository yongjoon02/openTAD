import json
import matplotlib.pyplot as plt
import pandas as pd
import time
import re
from pathlib import Path

def parse_log_file(log_file_path):
    """로그 파일에서 학습 메트릭 추출"""
    train_losses = []
    val_losses = []
    epochs = []
    learning_rates = []
    iteration_count = 0
    
    with open(log_file_path, 'r') as f:
        for line in f:
            if '[Train]' in line and 'Loss=' in line:
                # 훈련 손실 추출
                try:
                    # 정규표현식으로 더 정확한 파싱
                    loss_match = re.search(r'Loss=([\d.]+)', line)
                    if loss_match:
                        loss = float(loss_match.group(1))
                        train_losses.append(loss)
                        
                        # 에폭 추출 - [036] 형태에서 추출
                        epoch_match = re.search(r'\[(\d+)\]', line)
                        if epoch_match:
                            epoch = int(epoch_match.group(1))
                            epochs.append(epoch)
                        else:
                            # 에폭을 찾지 못한 경우 iteration_count 사용
                            epochs.append(iteration_count)
                        
                        # 학습률 추출
                        lr_match = re.search(r'lr_det=([\d.e+-]+)', line)
                        if lr_match:
                            lr = float(lr_match.group(1))
                            learning_rates.append(lr)
                        
                        iteration_count += 1
                        
                except Exception as e:
                    print(f"파싱 오류: {e}, 라인: {line.strip()}")
                    continue
                    
            elif '[Val]' in line and 'Loss' in line:
                # 검증 손실 추출
                try:
                    loss_match = re.search(r'Loss=([\d.]+)', line)
                    if loss_match:
                        val_loss = float(loss_match.group(1))
                        val_losses.append(val_loss)
                except:
                    continue
    
    # epochs가 비어있으면 인덱스로 대체
    if not epochs and train_losses:
        epochs = list(range(len(train_losses)))
    
    print(f"파싱 결과:")
    print(f"  훈련 손실 개수: {len(train_losses)}")
    print(f"  검증 손실 개수: {len(val_losses)}")
    print(f"  에폭 개수: {len(epochs)}")
    print(f"  학습률 개수: {len(learning_rates)}")
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates
    }

def plot_training_curves(data, save_path=None):
    """학습 곡선 플롯"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 훈련 손실
    if data['train_losses']:
        x_data = data['epochs'][:len(data['train_losses'])] if data['epochs'] else range(len(data['train_losses']))
        axes[0, 0].plot(x_data, data['train_losses'], 'b-', label='Train Loss', alpha=0.7)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch/Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 이동평균 추가
        if len(data['train_losses']) > 10:
            window = min(10, len(data['train_losses']) // 10)
            moving_avg = pd.Series(data['train_losses']).rolling(window=window).mean()
            axes[0, 0].plot(x_data, moving_avg, 'r-', label=f'Moving Avg (window={window})', linewidth=2)
            axes[0, 0].legend()
    
    # 검증 손실
    if data['val_losses']:
        x_data = range(len(data['val_losses']))
        axes[0, 1].plot(x_data, data['val_losses'], 'r-', label='Val Loss', alpha=0.7)
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 학습률
    if data['learning_rates']:
        x_data = range(len(data['learning_rates']))
        axes[1, 0].plot(x_data, data['learning_rates'], 'g-', label='Learning Rate', alpha=0.7)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')  # 로그 스케일
    
    # 훈련 vs 검증 손실 비교
    if data['train_losses'] and data['val_losses']:
        min_len = min(len(data['train_losses']), len(data['val_losses']))
        x_data = range(min_len)
        axes[1, 1].plot(x_data, data['train_losses'][:min_len], 'b-', label='Train Loss', alpha=0.7)
        axes[1, 1].plot(x_data, data['val_losses'][:min_len], 'r-', label='Val Loss', alpha=0.7)
        axes[1, 1].set_title('Train vs Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"학습 곡선이 {save_path}에 저장되었습니다.")
    
    plt.show()

def monitor_training(log_file_path, update_interval=60):
    """실시간 학습 모니터링"""
    print(f"학습 모니터링 시작: {log_file_path}")
    print("Ctrl+C로 중단할 수 있습니다.")
    
    last_size = 0
    
    try:
        while True:
            # 파일 크기 확인
            current_size = Path(log_file_path).stat().st_size
            
            if current_size > last_size:
                # 새로운 로그가 있으면 파싱
                data = parse_log_file(log_file_path)
                
                if data['train_losses']:
                    print(f"\n현재 상태:")
                    print(f"  에폭: {data['epochs'][-1] if data['epochs'] else 'N/A'}")
                    print(f"  최신 훈련 손실: {data['train_losses'][-1]:.4f}")
                    print(f"  최신 검증 손실: {data['val_losses'][-1] if data['val_losses'] else 'N/A'}")
                    print(f"  최신 학습률: {data['learning_rates'][-1] if data['learning_rates'] else 'N/A'}")
                
                last_size = current_size
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n모니터링 중단됨")
        
        # 최종 학습 곡선 생성
        data = parse_log_file(log_file_path)
        plot_training_curves(data, 'training_curves.png')

def analyze_training_progress(data):
    """학습 진행 상황 분석"""
    if not data['train_losses']:
        print("훈련 데이터가 없습니다.")
        return
    
    print("\n=== 학습 진행 상황 분석 ===")
    
    # 기본 통계
    train_losses = data['train_losses']
    print(f"총 훈련 반복: {len(train_losses)}")
    print(f"초기 손실: {train_losses[0]:.4f}")
    print(f"최신 손실: {train_losses[-1]:.4f}")
    print(f"손실 변화: {train_losses[0] - train_losses[-1]:.4f}")
    
    # 최근 10개 평균
    recent_avg = sum(train_losses[-10:]) / min(10, len(train_losses))
    print(f"최근 10개 평균 손실: {recent_avg:.4f}")
    
    # 학습률 분석
    if data['learning_rates']:
        lrs = data['learning_rates']
        print(f"초기 학습률: {lrs[0]:.2e}")
        print(f"현재 학습률: {lrs[-1]:.2e}")
    
    # 검증 손실 분석
    if data['val_losses']:
        val_losses = data['val_losses']
        print(f"검증 손실 개수: {len(val_losses)}")
        print(f"최신 검증 손실: {val_losses[-1]:.4f}")
        
        # 과적합 체크
        if len(train_losses) > 0 and len(val_losses) > 0:
            train_recent = train_losses[-1]
            val_recent = val_losses[-1]
            gap = val_recent - train_recent
            print(f"훈련-검증 손실 차이: {gap:.4f}")
            
            if gap > 0.1:
                print("⚠️ 과적합 가능성 (검증 손실이 훈련 손실보다 높음)")
            else:
                print("✅ 훈련-검증 손실 차이가 적절함")

# 사용 예시
log_file = r"F:\OpenTAD\work_dirs\e2e_pku_mmd_videomae_s_768x1_160_adapter_unfreeze\gpu1_id0\log.json"

# 현재까지의 학습 곡선 확인
print("로그 파일 파싱 중...")
data = parse_log_file(log_file)

# 학습 진행 상황 분석
analyze_training_progress(data)

# 학습 곡선 플롯
if data['train_losses']:
    plot_training_curves(data, 'current_training_curves.png')
else:
    print("훈련 데이터를 찾을 수 없습니다.")

# 실시간 모니터링 (선택사항)
# monitor_training(log_file, update_interval=30)