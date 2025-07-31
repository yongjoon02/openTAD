#!/usr/bin/env python3

import torch
import json
from pathlib import Path

def analyze_model_differences():
    """우리 모델과 official model의 차이점을 분석합니다"""
    
    print("🔍 모델 차이점 분석 시작...")
    print("=" * 60)
    
    # 1. 우리 모델 checkpoint 분석
    our_checkpoint_path = "exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter/gpu1_id0/checkpoint/epoch_59.pth"
    our_checkpoint = torch.load(our_checkpoint_path, map_location="cpu")
    
    # 2. Official model checkpoint 분석
    official_checkpoint_path = "pretrained/official/adatad_videomae_s_official_fixed.pth"
    official_checkpoint = torch.load(official_checkpoint_path, map_location="cpu")
    
    print("📊 Checkpoint 구조 비교:")
    print(f"   우리 모델 keys: {len(our_checkpoint.keys())}")
    print(f"   Official 모델 keys: {len(official_checkpoint.keys())}")
    
    # 3. State dict 비교
    our_state_dict = our_checkpoint["state_dict_ema"]
    official_state_dict = official_checkpoint["state_dict_ema"]
    
    print(f"\n🔧 State dict 비교:")
    print(f"   우리 모델 state dict keys: {len(our_state_dict.keys())}")
    print(f"   Official 모델 state dict keys: {len(official_state_dict.keys())}")
    
    # 4. Key 구조 분석
    our_keys = set(our_state_dict.keys())
    official_keys = set(official_state_dict.keys())
    
    print(f"\n📋 Key 구조 분석:")
    print(f"   공통 keys: {len(our_keys & official_keys)}")
    print(f"   우리만 있는 keys: {len(our_keys - official_keys)}")
    print(f"   Official만 있는 keys: {len(official_keys - our_keys)}")
    
    # 5. 우리만 있는 keys 확인
    if our_keys - official_keys:
        print(f"\n❓ 우리 모델에만 있는 keys (처음 10개):")
        for key in list(our_keys - official_keys)[:10]:
            print(f"   - {key}")
    
    # 6. Official만 있는 keys 확인
    if official_keys - our_keys:
        print(f"\n❓ Official 모델에만 있는 keys (처음 10개):")
        for key in list(official_keys - our_keys)[:10]:
            print(f"   - {key}")
    
    # 7. 가중치 값 비교 (공통 keys)
    common_keys = our_keys & official_keys
    print(f"\n🔍 가중치 값 비교 (공통 keys 중 샘플):")
    
    sample_keys = list(common_keys)[:5]
    for key in sample_keys:
        our_weight = our_state_dict[key]
        official_weight = official_state_dict[key]
        
        print(f"\n   Key: {key}")
        print(f"   우리 모델 shape: {our_weight.shape}")
        print(f"   Official 모델 shape: {official_weight.shape}")
        
        if our_weight.shape == official_weight.shape:
            diff = torch.abs(our_weight - official_weight).mean().item()
            print(f"   평균 차이: {diff:.6f}")
            
            if diff < 1e-6:
                print(f"   ✅ 거의 동일 (차이 < 1e-6)")
            elif diff < 1e-3:
                print(f"   ⚠️  약간 차이 (차이 < 1e-3)")
            else:
                print(f"   ❌ 큰 차이 (차이 >= 1e-3)")
    
    # 8. Training info 비교
    print(f"\n📈 Training 정보 비교:")
    
    if "meta" in our_checkpoint:
        print(f"   우리 모델 meta: {our_checkpoint['meta']}")
    
    if "meta" in official_checkpoint:
        print(f"   Official 모델 meta: {official_checkpoint['meta']}")
    
    # 9. Optimizer state 비교
    print(f"\n⚙️ Optimizer state 비교:")
    print(f"   우리 모델 optimizer keys: {len(our_checkpoint.get('optimizer', {}))}")
    print(f"   Official 모델 optimizer keys: {len(official_checkpoint.get('optimizer', {}))}")
    
    # 10. EMA state 비교
    print(f"\n📊 EMA state 비교:")
    print(f"   우리 모델 EMA: {'state_dict_ema' in our_checkpoint}")
    print(f"   Official 모델 EMA: {'state_dict_ema' in official_checkpoint}")
    
    # 11. 성능 차이 요약
    print(f"\n🏆 성능 차이 요약:")
    print(f"   우리 모델: 29.25% mAP")
    print(f"   Official 모델: 31.25% mAP")
    print(f"   차이: {31.25 - 29.25:.2f}%")
    
    return {
        "our_keys": len(our_keys),
        "official_keys": len(official_keys),
        "common_keys": len(common_keys),
        "our_only": len(our_keys - official_keys),
        "official_only": len(official_keys - our_keys)
    }

if __name__ == "__main__":
    analyze_model_differences() 