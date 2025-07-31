import torch
import numpy as np

def check_checkpoint(checkpoint_path):
    """체크포인트 가중치를 검증하는 함수"""
    print(f"체크포인트 검증: {checkpoint_path}")
    print("=" * 50)
    
    # 체크포인트 로드
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ 체크포인트 로드 성공")
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        return
    
    # 기본 정보 확인
    print(f"체크포인트 키: {list(checkpoint.keys())}")
    print(f"에폭: {checkpoint.get('epoch', 'Not found')}")
    
    # State dict 확인
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"\n📊 모델 파라미터 정보:")
        print(f"총 파라미터 수: {len(state_dict)}")
        
        # 파라미터 통계
        total_params = 0
        param_stats = []
        
        for key, param in state_dict.items():
            if param.dtype in [torch.float32, torch.float16]:
                total_params += param.numel()
                param_stats.append({
                    'key': key,
                    'shape': param.shape,
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'min': param.min().item(),
                    'max': param.max().item(),
                    'has_nan': torch.isnan(param).any().item(),
                    'has_inf': torch.isinf(param).any().item()
                })
        
        print(f"총 파라미터 개수: {total_params:,}")
        
        # 문제가 있는 파라미터 확인
        nan_params = [p for p in param_stats if p['has_nan']]
        inf_params = [p for p in param_stats if p['has_inf']]
        zero_params = [p for p in param_stats if p['std'] == 0]
        
        print(f"\n⚠️  문제가 있는 파라미터:")
        print(f"NaN 포함: {len(nan_params)}개")
        print(f"Inf 포함: {len(inf_params)}개")
        print(f"표준편차 0: {len(zero_params)}개")
        
        if nan_params:
            print("NaN이 포함된 파라미터 (처음 5개):")
            for p in nan_params[:5]:
                print(f"  {p['key']}: shape={p['shape']}")
        
        if inf_params:
            print("Inf가 포함된 파라미터 (처음 5개):")
            for p in inf_params[:5]:
                print(f"  {p['key']}: shape={p['shape']}")
        
        # 첫 번째 파라미터들의 상세 정보
        print(f"\n📋 첫 번째 파라미터들의 상세 정보:")
        for i, p in enumerate(param_stats[:10]):
            print(f"{i+1}. {p['key']}")
            print(f"   Shape: {p['shape']}")
            print(f"   Mean: {p['mean']:.6f}, Std: {p['std']:.6f}")
            print(f"   Range: [{p['min']:.6f}, {p['max']:.6f}]")
            print(f"   Has NaN: {p['has_nan']}, Has Inf: {p['has_inf']}")
            print()
    
    # Optimizer 상태 확인
    if 'optimizer' in checkpoint:
        optimizer = checkpoint['optimizer']
        print(f"\n🔧 Optimizer 정보:")
        print(f"Optimizer 키: {list(optimizer.keys())}")
        if 'state' in optimizer:
            print(f"Optimizer state 항목 수: {len(optimizer['state'])}")
    
    # EMA 상태 확인
    if 'state_dict_ema' in checkpoint:
        ema_state_dict = checkpoint['state_dict_ema']
        print(f"\n📈 EMA 정보:")
        print(f"EMA 파라미터 수: {len(ema_state_dict)}")
        
        # EMA 파라미터 통계
        ema_has_nan = any(torch.isnan(param).any() for param in ema_state_dict.values())
        ema_has_inf = any(torch.isinf(param).any() for param in ema_state_dict.values())
        print(f"EMA Has NaN: {ema_has_nan}")
        print(f"EMA Has Inf: {ema_has_inf}")
    
    print("\n" + "=" * 50)
    print("체크포인트 검증 완료")

if __name__ == "__main__":
    # 최신 체크포인트 검증
    checkpoint_path = "work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0/checkpoint/epoch_59.pth"
    check_checkpoint(checkpoint_path)
    
    # 이전 체크포인트와 비교 (선택사항)
    print("\n" + "=" * 50)
    print("이전 체크포인트와 비교")
    print("=" * 50)
    
    try:
        prev_checkpoint = torch.load("work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0/checkpoint/epoch_57.pth", map_location='cpu')
        current_checkpoint = torch.load("work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0/checkpoint/epoch_59.pth", map_location='cpu')
        
        print(f"이전 에폭: {prev_checkpoint.get('epoch', 'Not found')}")
        print(f"현재 에폭: {current_checkpoint.get('epoch', 'Not found')}")
        
        # 파라미터 변화 확인
        prev_state = prev_checkpoint['state_dict']
        curr_state = current_checkpoint['state_dict']
        
        print(f"\n파라미터 변화 분석:")
        total_diff = 0
        for key in curr_state.keys():
            if key in prev_state:
                diff = torch.abs(curr_state[key] - prev_state[key]).mean().item()
                total_diff += diff
        
        print(f"평균 파라미터 변화: {total_diff / len(curr_state):.8f}")
        
        if total_diff < 1e-8:
            print("⚠️  경고: 파라미터 변화가 매우 작습니다. 학습이 제대로 되지 않았을 수 있습니다.")
        elif total_diff > 1.0:
            print("⚠️  경고: 파라미터 변화가 매우 큽니다. 학습이 불안정할 수 있습니다.")
        else:
            print("✅ 파라미터 변화가 정상 범위입니다.")
            
    except Exception as e:
        print(f"이전 체크포인트 비교 실패: {e}") 