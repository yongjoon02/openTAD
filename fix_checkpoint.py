#!/usr/bin/env python3
"""
DataParallel로 저장된 체크포인트에서 'module.' prefix를 제거하는 스크립트
"""

import torch
import argparse

def fix_checkpoint(input_path, output_path):
    """
    체크포인트에서 'module.' prefix를 제거합니다.
    """
    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # state_dict에서 'module.' prefix 제거
    if 'state_dict_ema' in checkpoint:
        # EMA 버전이 있는 경우
        old_state_dict = checkpoint['state_dict_ema']
        new_state_dict = {}
        
        for key, value in old_state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 'module.' 제거 (7글자)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        checkpoint['state_dict_ema'] = new_state_dict
        print(f"✅ Fixed {len(new_state_dict)} keys in state_dict_ema")
    
    if 'state_dict' in checkpoint:
        # 일반 state_dict도 처리
        old_state_dict = checkpoint['state_dict']
        new_state_dict = {}
        
        for key, value in old_state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 'module.' 제거
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        checkpoint['state_dict'] = new_state_dict
        print(f"✅ Fixed {len(new_state_dict)} keys in state_dict")
    
    # 수정된 체크포인트 저장
    print(f"Saving fixed checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("✅ Checkpoint fix completed!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input checkpoint path')
    parser.add_argument('--output', required=True, help='Output checkpoint path')
    
    args = parser.parse_args()
    fix_checkpoint(args.input, args.output)

if __name__ == '__main__':
    main() 