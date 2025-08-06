#!/usr/bin/env python3
"""
상세 데이터로더 테스트 스크립트
영상과 어노테이션 매칭을 단계별로 확인
"""

import sys
import os
sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), ".")
if path not in sys.path:
    sys.path.insert(0, path)

import torch
import numpy as np
import json
from mmengine.config import Config
from opentad.datasets import build_dataset, build_dataloader
from opentad.utils import setup_logger


def load_original_annotations(annotation_file):
    """원본 어노테이션 파일 로드"""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # 비디오별로 어노테이션 정리
    video_annotations = {}
    for anno in annotations:
        video_name = anno['video_name']
        if video_name not in video_annotations:
            video_annotations[video_name] = []
        video_annotations[video_name].append(anno)
    
    return video_annotations


def test_dataloader_detailed(config_path, num_samples=2):
    """상세 데이터로더 테스트 함수"""
    
    print("=" * 80)
    print("🔍 상세 데이터로더 테스트 시작")
    print("=" * 80)
    
    # 설정 로드
    cfg = Config.fromfile(config_path)
    print(f"📁 설정 파일: {config_path}")
    
    # 원본 어노테이션 로드
    print("\n📖 원본 어노테이션 로드 중...")
    train_annotations = load_original_annotations(cfg.dataset.train.ann_file)
    val_annotations = load_original_annotations(cfg.dataset.val.ann_file)
    
    print(f"✅ 훈련 원본 어노테이션: {len(train_annotations)} 비디오")
    print(f"✅ 검증 원본 어노테이션: {len(val_annotations)} 비디오")
    
    # 로거 설정
    logger = setup_logger("TestDataloader", save_dir=None, distributed_rank=0)
    
    # 데이터셋 빌드
    print("\n📊 데이터셋 빌드 중...")
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    
    print(f"✅ 훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"✅ 검증 데이터셋 크기: {len(val_dataset)}")
    
    # 메모리 절약을 위한 설정
    train_config = cfg.solver.train.copy()
    val_config = cfg.solver.val.copy()
    
    train_config['batch_size'] = 1
    val_config['batch_size'] = 1
    train_config['num_workers'] = 0
    val_config['num_workers'] = 0
    
    # 데이터로더 빌드
    print("\n🔄 데이터로더 빌드 중...")
    train_loader = build_dataloader(
        train_dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **train_config,
    )
    
    val_loader = build_dataloader(
        val_dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **val_config,
    )
    
    # 훈련 데이터 테스트
    print("\n" + "=" * 50)
    print("🎬 훈련 데이터 상세 테스트")
    print("=" * 50)
    
    train_iter = iter(train_loader)
    for i in range(min(num_samples, len(train_loader))):
        try:
            batch = next(train_iter)
            print(f"\n📦 배치 {i+1}:")
            
            # 배치 키 구조 확인
            print(f"  🔑 배치 키: {list(batch.keys())}")
            
            # 이미지 형태 확인
            if 'inputs' in batch:
                print(f"  🖼️  이미지 형태: {batch['inputs'].shape}")
            
            # 메타 정보 상세 확인
            if 'metas' in batch:
                print(f"  📋 메타 정보:")
                for j, meta in enumerate(batch['metas']):
                    print(f"    📺 샘플 {j} 메타:")
                    for key, value in meta.items():
                        if key == 'video_name':
                            print(f"      - 영상 제목: {value}")
                            
                            # 원본 어노테이션과 비교
                            if value in train_annotations:
                                original_annos = train_annotations[value]
                                print(f"      - 원본 어노테이션 개수: {len(original_annos)}")
                                for anno_idx, anno in enumerate(original_annos):
                                    print(f"        원본 {anno_idx}: {anno['segment']} -> {anno['label']}")
                            else:
                                print(f"      - ⚠️ 원본 어노테이션에서 찾을 수 없음")
                                
                        elif key == 'frame_inds':
                            print(f"      - 프레임 인덱스 형태: {value.shape if hasattr(value, 'shape') else type(value)}")
                            if hasattr(value, 'shape'):
                                print(f"      - 프레임 인덱스 범위: {value.min().item()} ~ {value.max().item()}")
                        elif key == 'masks':
                            print(f"      - 마스크 형태: {value.shape if hasattr(value, 'shape') else type(value)}")
                            if hasattr(value, 'shape'):
                                valid_count = value.sum().item()
                                print(f"      - 유효 프레임 수: {valid_count}")
                        else:
                            print(f"      - {key}: {value}")
            
            # 어노테이션 정보 상세 확인
            if 'gt_segments' in batch:
                print(f"  🎯 처리된 어노테이션:")
                for j, segments in enumerate(batch['gt_segments']):
                    if len(segments) > 0:
                        print(f"    📺 샘플 {j}:")
                        print(f"      - 세그먼트: {segments.tolist()}")
                        
                        if 'gt_labels' in batch and j < len(batch['gt_labels']):
                            labels = batch['gt_labels'][j]
                            print(f"      - 라벨: {labels.tolist()}")
                        
                        # 프레임 범위와 매칭 확인
                        if 'frame_inds' in batch and 'masks' in batch and j < len(batch['frame_inds']) and j < len(batch['masks']):
                            frame_inds = batch['frame_inds'][j]
                            masks = batch['masks'][j]
                            valid_frames = frame_inds[masks]
                            
                            if len(valid_frames) > 0:
                                print(f"      - 유효 프레임 범위: {valid_frames[0].item()} ~ {valid_frames[-1].item()}")
                                print(f"      - 총 유효 프레임: {len(valid_frames)}")
                                
                                # 어노테이션이 유효 프레임 범위 내에 있는지 확인
                                frame_start = valid_frames[0].item()
                                frame_end = valid_frames[-1].item()
                                
                                for seg_idx, (start, end) in enumerate(segments):
                                    if start >= frame_start and end <= frame_end:
                                        print(f"      ✅ 세그먼트 {seg_idx}: 프레임 범위 내 ({start:.1f} ~ {end:.1f})")
                                    else:
                                        print(f"      ⚠️  세그먼트 {seg_idx}: 프레임 범위 벗어남 ({start:.1f} ~ {end:.1f})")
                                        print(f"         프레임 범위: {frame_start} ~ {frame_end}")
                    else:
                        print(f"    📺 샘플 {j}: 어노테이션 없음")
            
            # 메모리 정리
            del batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"❌ 배치 {i+1} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 검증 데이터 테스트
    print("\n" + "=" * 50)
    print("🔍 검증 데이터 상세 테스트")
    print("=" * 50)
    
    val_iter = iter(val_loader)
    for i in range(min(num_samples, len(val_loader))):
        try:
            batch = next(val_iter)
            print(f"\n📦 배치 {i+1}:")
            
            # 배치 키 구조 확인
            print(f"  🔑 배치 키: {list(batch.keys())}")
            
            # 이미지 형태 확인
            if 'inputs' in batch:
                print(f"  🖼️  이미지 형태: {batch['inputs'].shape}")
            
            # 메타 정보 상세 확인
            if 'metas' in batch:
                print(f"  📋 메타 정보:")
                for j, meta in enumerate(batch['metas']):
                    print(f"    📺 샘플 {j} 메타:")
                    for key, value in meta.items():
                        if key == 'video_name':
                            print(f"      - 영상 제목: {value}")
                            
                            # 원본 어노테이션과 비교
                            if value in val_annotations:
                                original_annos = val_annotations[value]
                                print(f"      - 원본 어노테이션 개수: {len(original_annos)}")
                                for anno_idx, anno in enumerate(original_annos):
                                    print(f"        원본 {anno_idx}: {anno['segment']} -> {anno['label']}")
                            else:
                                print(f"      - ⚠️ 원본 어노테이션에서 찾을 수 없음")
                                
                        elif key == 'frame_inds':
                            print(f"      - 프레임 인덱스 형태: {value.shape if hasattr(value, 'shape') else type(value)}")
                            if hasattr(value, 'shape'):
                                print(f"      - 프레임 인덱스 범위: {value.min().item()} ~ {value.max().item()}")
                        elif key == 'masks':
                            print(f"      - 마스크 형태: {value.shape if hasattr(value, 'shape') else type(value)}")
                            if hasattr(value, 'shape'):
                                valid_count = value.sum().item()
                                print(f"      - 유효 프레임 수: {valid_count}")
                        else:
                            print(f"      - {key}: {value}")
            
            # 어노테이션 정보 상세 확인
            if 'gt_segments' in batch:
                print(f"  🎯 처리된 어노테이션:")
                for j, segments in enumerate(batch['gt_segments']):
                    if len(segments) > 0:
                        print(f"    📺 샘플 {j}:")
                        print(f"      - 세그먼트: {segments.tolist()}")
                        
                        if 'gt_labels' in batch and j < len(batch['gt_labels']):
                            labels = batch['gt_labels'][j]
                            print(f"      - 라벨: {labels.tolist()}")
                        
                        # 프레임 범위와 매칭 확인
                        if 'frame_inds' in batch and 'masks' in batch and j < len(batch['frame_inds']) and j < len(batch['masks']):
                            frame_inds = batch['frame_inds'][j]
                            masks = batch['masks'][j]
                            valid_frames = frame_inds[masks]
                            
                            if len(valid_frames) > 0:
                                print(f"      - 유효 프레임 범위: {valid_frames[0].item()} ~ {valid_frames[-1].item()}")
                                print(f"      - 총 유효 프레임: {len(valid_frames)}")
                                
                                # 어노테이션이 유효 프레임 범위 내에 있는지 확인
                                frame_start = valid_frames[0].item()
                                frame_end = valid_frames[-1].item()
                                
                                for seg_idx, (start, end) in enumerate(segments):
                                    if start >= frame_start and end <= frame_end:
                                        print(f"      ✅ 세그먼트 {seg_idx}: 프레임 범위 내 ({start:.1f} ~ {end:.1f})")
                                    else:
                                        print(f"      ⚠️  세그먼트 {seg_idx}: 프레임 범위 벗어남 ({start:.1f} ~ {end:.1f})")
                                        print(f"         프레임 범위: {frame_start} ~ {frame_end}")
                    else:
                        print(f"    📺 샘플 {j}: 어노테이션 없음")
            
            # 메모리 정리
            del batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"❌ 배치 {i+1} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("✅ 상세 데이터로더 테스트 완료!")
    print("=" * 80)
    
    # 설정 요약
    print("\n📋 현재 설정 요약:")
    print(f"  - feature_stride: {cfg.dataset.train.get('feature_stride', 'N/A')}")
    print(f"  - sample_stride: {cfg.dataset.train.get('sample_stride', 'N/A')}")
    print(f"  - snippet_stride: {cfg.dataset.train.get('feature_stride', 1) * cfg.dataset.train.get('sample_stride', 1)}")
    print(f"  - scale_factor: {cfg.dataset.train.pipeline[2].get('scale_factor', 'N/A')}")
    print(f"  - FPS: 30.0")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="상세 데이터로더 테스트")
    parser.add_argument("config", type=str, help="설정 파일 경로")
    parser.add_argument("--num_samples", type=int, default=2, help="테스트할 배치 수")
    
    args = parser.parse_args()
    
    try:
        test_dataloader_detailed(args.config, args.num_samples)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc() 