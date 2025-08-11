#!/usr/bin/env python3
"""
프레임 범위 변환 테스트 스크립트
PKU-MMD에서 윈도우 기반 프레임 범위가 올바르게 복원되는지 확인
"""

import numpy as np
import torch

def test_frame_conversion():
    """프레임 범위 변환 테스트"""
    print("=" * 60)
    print("🔍 프레임 범위 변환 테스트")
    print("=" * 60)
    
    # 테스트 케이스
    test_cases = [
        {
            "video_name": "0005-L",
            "original_segment": [229, 293],  # 전체 비디오 기준
            "window_start_frame": 200,       # 윈도우 시작점
            "snippet_stride": 1,             # snippet stride
            "offset_frames": 0,              # offset
            "fps": 30.0                      # PKU-MMD FPS
        },
        {
            "video_name": "0005-M", 
            "original_segment": [377, 423],
            "window_start_frame": 350,
            "snippet_stride": 1,
            "offset_frames": 0,
            "fps": 30.0
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n📺 테스트 케이스 {i+1}: {case['video_name']}")
        print(f"  원본 세그먼트: {case['original_segment']}")
        print(f"  윈도우 시작점: {case['window_start_frame']}")
        
        # 1. 데이터로더 변환 (전체 -> 윈도우 기준)
        original_segments = np.array(case['original_segment'], dtype=np.float32)
        window_start = case['window_start_frame']
        snippet_stride = case['snippet_stride']
        offset_frames = case['offset_frames']
        
        # 데이터로더에서 하는 변환
        window_segments = (
            original_segments - window_start - offset_frames
        ) / snippet_stride
        
        print(f"  윈도우 기준 세그먼트: {window_segments}")
        
        # 2. 모델 예측 (윈도우 기준)
        # 실제로는 모델이 예측하지만, 여기서는 테스트용으로 약간 수정된 값 사용
        predicted_segments = window_segments + np.array([-4, 8])  # 약간의 오차
        print(f"  모델 예측 (윈도우 기준): {predicted_segments}")
        
        # 3. 평가 시 복원 (윈도우 기준 -> 전체 기준)
        # 기존 방식 (잘못된 방식)
        old_restored = (
            predicted_segments * snippet_stride + window_start + offset_frames
        ) / case['fps']  # fps로 나누면 초 단위가 됨!
        
        # 수정된 방식 (올바른 방식)
        new_restored = (
            predicted_segments * snippet_stride + window_start + offset_frames
        )  # fps로 나누지 않음 (프레임 단위 유지)
        
        print(f"  기존 복원 방식 (초 단위): {old_restored}")
        print(f"  수정된 복원 방식 (프레임 단위): {new_restored}")
        print(f"  원본과의 차이: {np.abs(new_restored - case['original_segment'])}")
        
        # 4. 정확도 확인
        original = np.array(case['original_segment'])
        error = np.abs(new_restored - original)
        accuracy = 1.0 - (error.sum() / (original[1] - original[0]))
        
        print(f"  복원 정확도: {accuracy:.3f}")
        
        if accuracy > 0.9:
            print(f"  ✅ 정확한 복원!")
        else:
            print(f"  ❌ 복원 오류!")

def test_evaluation_conversion():
    """평가 시 변환 로직 테스트"""
    print("\n" + "=" * 60)
    print("🔍 평가 시 변환 로직 테스트")
    print("=" * 60)
    
    # 실제 평가에서 사용되는 변환 로직
    def convert_to_original_frame(segments, meta):
        """평가 시 윈도우 기준 -> 전체 기준 변환"""
        snippet_stride = meta["snippet_stride"]
        offset_frames = meta["offset_frames"]
        window_start_frame = meta.get("window_start_frame", 0)
        fps = meta.get("fps", 30.0)
        
        # PKU-MMD 특별 처리: 프레임 단위 유지
        if fps == 30.0:  # PKU-MMD
            restored = segments * snippet_stride + window_start_frame + offset_frames
        else:  # 다른 데이터셋 (초 단위로 변환)
            restored = (segments * snippet_stride + window_start_frame + offset_frames) / fps
        
        return restored
    
    # 테스트 메타데이터
    meta = {
        "snippet_stride": 1,
        "offset_frames": 0,
        "window_start_frame": 200,
        "fps": 30.0
    }
    
    # 테스트 세그먼트 (윈도우 기준)
    window_segments = np.array([[29, 93], [177, 223]], dtype=np.float32)
    
    print(f"윈도우 기준 세그먼트: {window_segments}")
    print(f"메타데이터: {meta}")
    
    # 변환
    original_segments = convert_to_original_frame(window_segments, meta)
    
    print(f"전체 기준 세그먼트: {original_segments}")
    
    # 검증
    expected = np.array([[229, 293], [377, 423]])  # 예상 결과
    error = np.abs(original_segments - expected)
    
    print(f"예상 결과: {expected}")
    print(f"오차: {error}")
    
    if np.all(error < 1.0):
        print("✅ 변환 정확!")
    else:
        print("❌ 변환 오류!")

if __name__ == "__main__":
    test_frame_conversion()
    test_evaluation_conversion() 