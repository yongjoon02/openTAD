import json
import re

def check_training_log():
    """로그 파일의 실제 형식을 확인하고 학습 정보를 찾음"""
    
    log_file = "work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0/log.json"
    
    print("=== 로그 파일 형식 분석 ===\n")
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        print(f"총 로그 라인 수: {len(lines):,}")
        
        # 로그에서 다양한 패턴 찾기
        patterns = {
            'loss': [],
            'epoch': [],
            'train': [],
            'val': [],
            'lr': [],
            'accuracy': [],
            'mAP': []
        }
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            if 'loss' in line_lower:
                patterns['loss'].append((i, line.strip()))
            if 'epoch' in line_lower:
                patterns['epoch'].append((i, line.strip()))
            if 'train' in line_lower:
                patterns['train'].append((i, line.strip()))
            if 'val' in line_lower:
                patterns['val'].append((i, line.strip()))
            if 'lr' in line_lower or 'learning_rate' in line_lower:
                patterns['lr'].append((i, line.strip()))
            if 'accuracy' in line_lower or 'acc' in line_lower:
                patterns['accuracy'].append((i, line.strip()))
            if 'map' in line_lower:
                patterns['mAP'].append((i, line.strip()))
        
        # 결과 출력
        for key, matches in patterns.items():
            print(f"{key.upper()} 관련 라인: {len(matches)}개")
            if len(matches) > 0:
                print(f"  첫 번째: {matches[0][1]}")
                if len(matches) > 1:
                    print(f"  마지막: {matches[-1][1]}")
            print()
        
        # 학습 관련 정보가 있는지 확인
        print("=== 학습 정보 확인 ===")
        
        # epoch 정보 찾기
        epoch_lines = [line for _, line in patterns['epoch']]
        if epoch_lines:
            print("Epoch 정보 발견:")
            for line in epoch_lines[:5]:  # 처음 5개만
                print(f"  {line}")
            if len(epoch_lines) > 5:
                print(f"  ... (총 {len(epoch_lines)}개)")
        
        # loss 정보 찾기 (다양한 패턴)
        loss_patterns = [
            r'loss[:\s]*([0-9.]+)',
            r'train_loss[:\s]*([0-9.]+)',
            r'val_loss[:\s]*([0-9.]+)',
            r'cls_loss[:\s]*([0-9.]+)',
            r'reg_loss[:\s]*([0-9.]+)'
        ]
        
        found_losses = []
        for line_num, line in patterns['loss']:
            for pattern in loss_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    found_losses.append((line_num, float(match.group(1)), line.strip()))
                    break
        
        if found_losses:
            print(f"\nLoss 값 발견: {len(found_losses)}개")
            print("처음 5개:")
            for line_num, loss_val, line in found_losses[:5]:
                print(f"  라인 {line_num}: {loss_val:.6f} - {line}")
            if len(found_losses) > 5:
                print(f"  ... (총 {len(found_losses)}개)")
            
            # Loss 통계
            loss_values = [loss for _, loss, _ in found_losses]
            print(f"\nLoss 통계:")
            print(f"  범위: {min(loss_values):.6f} ~ {max(loss_values):.6f}")
            print(f"  평균: {sum(loss_values)/len(loss_values):.6f}")
        else:
            print("\n❌ Loss 값을 찾을 수 없습니다.")
        
        # 학습 진행 상황 확인
        print("\n=== 학습 진행 상황 ===")
        
        # 시작과 끝 시간 확인
        start_time = None
        end_time = None
        
        for line in lines:
            if 'Train INFO' in line and 'Using torch version' in line:
                start_time = line.split('Train INFO:')[0].strip()
                break
        
        for line in reversed(lines):
            if 'Test INFO' in line and 'Testing Over' in line:
                end_time = line.split('Test INFO:')[0].strip()
                break
        
        if start_time and end_time:
            print(f"학습 시작: {start_time}")
            print(f"테스트 완료: {end_time}")
        
        # 체크포인트 정보 확인
        checkpoint_lines = [line for line in lines if 'checkpoint' in line.lower()]
        if checkpoint_lines:
            print(f"\n체크포인트 정보: {len(checkpoint_lines)}개")
            for line in checkpoint_lines[-3:]:  # 마지막 3개
                print(f"  {line.strip()}")
        
    except FileNotFoundError:
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
    except Exception as e:
        print(f"❌ 로그 파일 읽기 오류: {e}")

if __name__ == "__main__":
    check_training_log() 