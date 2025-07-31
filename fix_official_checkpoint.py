#!/usr/bin/env python3

import torch

def fix_official_checkpoint():
    """Official model의 'module.' prefix를 제거하여 single GPU에서 로드 가능하게 만듭니다"""
    
    print("🔧 Official checkpoint 수정 중...")
    
    # Official model 로드
    checkpoint_path = "pretrained/official/adatad_videomae_s_official.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    print(f"📁 원본 checkpoint keys 개수: {len(checkpoint['state_dict_ema'])}")
    
    # 새로운 state_dict 생성 (module. prefix 제거)
    new_state_dict = {}
    for key, value in checkpoint["state_dict_ema"].items():
        if key.startswith("module."):
            new_key = key[7:]  # "module." 제거 (7글자)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # 수정된 checkpoint 저장
    checkpoint["state_dict_ema"] = new_state_dict
    
    # 새로운 경로에 저장
    fixed_path = "pretrained/official/adatad_videomae_s_official_fixed.pth"
    torch.save(checkpoint, fixed_path)
    
    print(f"✅ 수정된 checkpoint 저장 완료: {fixed_path}")
    print(f"📁 수정된 checkpoint keys 개수: {len(new_state_dict)}")
    
    # 몇 개 key 예시 출력
    print("\n🔍 수정된 key 예시:")
    for i, key in enumerate(list(new_state_dict.keys())[:5]):
        print(f"  {i+1}. {key}")
    
    return fixed_path

if __name__ == "__main__":
    fix_official_checkpoint() 