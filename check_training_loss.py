import json
import re
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_loss():
    """학습 로그에서 loss 변화를 분석"""
    
    log_file = "work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0/log.json"
    
    print("=== 학습 Loss 분석 ===\n")
    
    # 로그 파일에서 loss 정보 추출
    train_losses = []
    val_losses = []
    epochs = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'Train' in line and 'loss' in line.lower():
                    # loss 값 추출
                    loss_match = re.search(r'loss[:\s]*([0-9.]+)', line, re.IGNORECASE)
                    if loss_match:
                        loss_value = float(loss_match.group(1))
                        train_losses.append(loss_value)
                        
                        # epoch 추출
                        epoch_match = re.search(r'epoch[:\s]*([0-9]+)', line, re.IGNORECASE)
                        if epoch_match:
                            epoch = int(epoch_match.group(1))
                            epochs.append(epoch)
                
                elif 'Val' in line and 'loss' in line.lower():
                    loss_match = re.search(r'loss[:\s]*([0-9.]+)', line, re.IGNORECASE)
                    if loss_match:
                        val_losses.append(float(loss_match.group(1)))
    
    except FileNotFoundError:
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
        return
    
    print(f"총 학습 loss 기록: {len(train_losses)}개")
    print(f"총 검증 loss 기록: {len(val_losses)}개")
    
    if len(train_losses) == 0:
        print("❌ 학습 loss 정보를 찾을 수 없습니다.")
        return
    
    # Loss 통계
    train_losses = np.array(train_losses)
    print(f"\n=== 학습 Loss 통계 ===")
    print(f"초기 loss: {train_losses[0]:.6f}")
    print(f"최종 loss: {train_losses[-1]:.6f}")
    print(f"최소 loss: {train_losses.min():.6f}")
    print(f"최대 loss: {train_losses.max():.6f}")
    print(f"Loss 감소량: {train_losses[0] - train_losses[-1]:.6f}")
    print(f"Loss 감소율: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
    
    # Loss 변화 분석
    print(f"\n=== Loss 변화 분석 ===")
    
    # 초기 10개 epoch
    if len(train_losses) >= 10:
        initial_avg = train_losses[:10].mean()
        final_avg = train_losses[-10:].mean()
        print(f"초기 10개 epoch 평균: {initial_avg:.6f}")
        print(f"최종 10개 epoch 평균: {final_avg:.6f}")
        print(f"전체 감소량: {initial_avg - final_avg:.6f}")
    
    # Loss 수렴 여부
    if len(train_losses) >= 20:
        recent_losses = train_losses[-20:]
        loss_std = recent_losses.std()
        loss_mean = recent_losses.mean()
        cv = loss_std / loss_mean  # 변동계수
        
        print(f"최근 20개 epoch 표준편차: {loss_std:.6f}")
        print(f"최근 20개 epoch 변동계수: {cv:.4f}")
        
        if cv < 0.1:
            print("✅ Loss가 안정적으로 수렴됨")
        elif cv < 0.3:
            print("⚠️ Loss가 어느 정도 수렴됨")
        else:
            print("❌ Loss가 불안정함")
    
    # Loss 증가 구간 확인
    increasing_epochs = []
    for i in range(1, len(train_losses)):
        if train_losses[i] > train_losses[i-1]:
            increasing_epochs.append(i)
    
    if increasing_epochs:
        print(f"\nLoss가 증가한 epoch 수: {len(increasing_epochs)}개")
        if len(increasing_epochs) > len(train_losses) * 0.3:
            print("⚠️ Loss가 자주 증가함 (학습 불안정)")
        else:
            print("✅ Loss 증가는 정상 범위")
    else:
        print("✅ Loss가 지속적으로 감소함")
    
    # 검증 loss 분석
    if len(val_losses) > 0:
        val_losses = np.array(val_losses)
        print(f"\n=== 검증 Loss 분석 ===")
        print(f"검증 loss 범위: {val_losses.min():.6f} ~ {val_losses.max():.6f}")
        print(f"검증 loss 평균: {val_losses.mean():.6f}")
        
        # 과적합 확인
        if len(train_losses) >= len(val_losses):
            recent_train = train_losses[-len(val_losses):]
            overfitting = val_losses.mean() - recent_train.mean()
            print(f"과적합 정도 (val - train): {overfitting:.6f}")
            
            if overfitting > 0.1:
                print("❌ 심각한 과적합")
            elif overfitting > 0.05:
                print("⚠️ 약간의 과적합")
            else:
                print("✅ 과적합 없음")
    
    # 학습 진행도 확인
    print(f"\n=== 학습 진행도 ===")
    if len(epochs) > 0:
        print(f"기록된 epoch 범위: {min(epochs)} ~ {max(epochs)}")
        if max(epochs) >= 50:
            print("✅ 충분한 epoch 학습됨")
        elif max(epochs) >= 30:
            print("⚠️ 중간 정도 학습됨")
        else:
            print("❌ 학습이 부족함")
    
    # 문제 진단
    print(f"\n=== 문제 진단 ===")
    
    if train_losses[-1] > train_losses[0] * 0.8:
        print("❌ Loss가 충분히 감소하지 않음")
    elif train_losses[-1] < train_losses[0] * 0.1:
        print("✅ Loss가 크게 감소함")
    else:
        print("⚠️ Loss가 어느 정도 감소함")
    
    if train_losses[-1] > 1.0:
        print("❌ 최종 loss가 너무 높음")
    elif train_losses[-1] > 0.5:
        print("⚠️ 최종 loss가 높음")
    else:
        print("✅ 최종 loss가 적절함")

if __name__ == "__main__":
    analyze_training_loss() 