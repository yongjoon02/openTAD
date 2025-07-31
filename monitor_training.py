#!/usr/bin/env python3
import time
import json
import re
from pathlib import Path

def monitor_training():
    log_path = Path("exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter/gpu1_id0/log.json")
    
    if not log_path.exists():
        print("❌ 로그 파일을 찾을 수 없습니다!")
        return
    
    print("🎯 AdaTAD 학습 모니터링 시작...")
    print("=" * 60)
    
    last_size = 0
    while True:
        try:
            current_size = log_path.stat().st_size
            if current_size > last_size:
                with open(log_path, 'r', encoding='utf-8') as f:
                    f.seek(last_size)
                    new_lines = f.readlines()
                
                for line in new_lines:
                    # Loss 정보 추출
                    if "Loss=" in line:
                        loss_match = re.search(r'Loss=([0-9.]+)', line)
                        cls_match = re.search(r'cls_loss=([0-9.]+)', line)
                        reg_match = re.search(r'reg_loss=([0-9.]+)', line)
                        epoch_match = re.search(r'\[(\d+)\]', line)
                        step_match = re.search(r'\[(\d+)/(\d+)\]', line)
                        
                        if all([loss_match, cls_match, reg_match, epoch_match, step_match]):
                            epoch = epoch_match.group(1)
                            step = step_match.group(1)
                            total_steps = step_match.group(2)
                            loss = float(loss_match.group(1))
                            cls_loss = float(cls_match.group(1))
                            reg_loss = float(reg_match.group(1))
                            
                            progress = int(step) / int(total_steps) * 100
                            
                            print(f"📊 Epoch {epoch} [{step}/{total_steps}] ({progress:.1f}%)")
                            print(f"   Loss: {loss:.4f} | cls: {cls_loss:.4f} | reg: {reg_loss:.4f}")
                            
                            # 성능 평가
                            if loss < 1.0:
                                print("   🎯 Loss < 1.0 - 좋은 성능!")
                            elif loss < 1.5:
                                print("   ✅ Loss < 1.5 - 순조로운 학습")
                    
                    # mAP 정보 추출 (Validation 시작 후)
                    if "mAP" in line:
                        map_match = re.search(r'mAP.*?([0-9.]+)', line)
                        if map_match:
                            map_value = map_match.group(1)
                            print(f"🏆 mAP: {map_value}%")
                    
                    # Epoch 시작/완료
                    if "Epoch" in line and "started" in line:
                        epoch_match = re.search(r'Epoch (\d+) started', line)
                        if epoch_match:
                            epoch = epoch_match.group(1)
                            print(f"🚀 Epoch {epoch} 시작!")
                
                last_size = current_size
            
            time.sleep(30)  # 30초마다 체크
            
        except KeyboardInterrupt:
            print("\n⏹️  모니터링 중단됨")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_training() 