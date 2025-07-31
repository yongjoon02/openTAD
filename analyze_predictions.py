import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_prediction_distributions():
    """예측 결과의 분포를 분석"""
    
    # 수정된 결과 파일 로드
    result_file = "work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0/result_detection_fixed.json"
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # 클래스 맵 로드 (라벨 이름 확인용)
    class_map_file = "data/PKU-MMD/class_map.txt"
    with open(class_map_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    print("=== 예측 결과 분포 분석 ===\n")
    
    # 데이터 수집
    all_labels = []
    all_segments = []
    all_scores = []
    all_segment_lengths = []
    
    for video_id, predictions in results['results'].items():
        for pred in predictions:
            all_labels.append(pred['label'])
            all_segments.extend(pred['segment'])
            all_scores.append(pred['score'])
            # segment 길이 계산
            segment_length = pred['segment'][1] - pred['segment'][0]
            all_segment_lengths.append(segment_length)
    
    total_predictions = len(all_scores)
    print(f"총 예측 수: {total_predictions:,}")
    
    # 1. 라벨 분포 분석
    print("\n=== 1. 라벨 분포 ===")
    label_counts = Counter(all_labels)
    print(f"사용된 라벨 수: {len(label_counts)} / 51")
    
    # 상위 10개 라벨
    print("\n상위 10개 라벨:")
    for label, count in label_counts.most_common(10):
        percentage = (count / total_predictions) * 100
        class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
        print(f"  라벨 {label} ({class_name}): {count:,}개 ({percentage:.2f}%)")
    
    # 하위 10개 라벨
    print("\n하위 10개 라벨:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1])[:10]:
        percentage = (count / total_predictions) * 100
        class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
        print(f"  라벨 {label} ({class_name}): {count:,}개 ({percentage:.2f}%)")
    
    # 라벨 분포 균등성
    expected_count = total_predictions / 51
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    print(f"\n라벨 분포 균등성:")
    print(f"  예상 개수 (균등 분포): {expected_count:.0f}")
    print(f"  최대 개수: {max_count:,}")
    print(f"  최소 개수: {min_count:,}")
    print(f"  최대/최소 비율: {max_count/min_count:.2f}")
    
    # 2. 스코어 분포 분석
    print("\n=== 2. 스코어 분포 ===")
    scores = np.array(all_scores)
    print(f"스코어 범위: {scores.min():.6f} ~ {scores.max():.6f}")
    print(f"스코어 평균: {scores.mean():.6f}")
    print(f"스코어 중앙값: {np.median(scores):.6f}")
    print(f"스코어 표준편차: {scores.std():.6f}")
    
    # 스코어 구간별 분포
    score_bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    print("\n스코어 구간별 분포:")
    for i in range(len(score_bins)-1):
        count = np.sum((scores >= score_bins[i]) & (scores < score_bins[i+1]))
        percentage = (count / total_predictions) * 100
        print(f"  {score_bins[i]:.2f}~{score_bins[i+1]:.2f}: {count:,}개 ({percentage:.2f}%)")
    
    # 3. Segment 분포 분석
    print("\n=== 3. Segment 분포 ===")
    segments = np.array(all_segments)
    segment_lengths = np.array(all_segment_lengths)
    
    print(f"Segment 시작점 범위: {segments[::2].min():.2f} ~ {segments[::2].max():.2f}")
    print(f"Segment 끝점 범위: {segments[1::2].min():.2f} ~ {segments[1::2].max():.2f}")
    print(f"Segment 길이 범위: {segment_lengths.min():.2f} ~ {segment_lengths.max():.2f}")
    print(f"Segment 길이 평균: {segment_lengths.mean():.2f}")
    print(f"Segment 길이 중앙값: {np.median(segment_lengths):.2f}")
    
    # Segment 길이 구간별 분포
    length_bins = [0, 10, 30, 50, 100, 200, 500, 1000, float('inf')]
    print("\nSegment 길이 구간별 분포:")
    for i in range(len(length_bins)-1):
        if length_bins[i+1] == float('inf'):
            count = np.sum(segment_lengths >= length_bins[i])
            label = f"{length_bins[i]:.0f}+"
        else:
            count = np.sum((segment_lengths >= length_bins[i]) & (segment_lengths < length_bins[i+1]))
            label = f"{length_bins[i]:.0f}~{length_bins[i+1]:.0f}"
        percentage = (count / total_predictions) * 100
        print(f"  {label}: {count:,}개 ({percentage:.2f}%)")
    
    # 4. 문제점 진단
    print("\n=== 4. 문제점 진단 ===")
    
    # 라벨 불균형 문제
    if max_count / min_count > 100:
        print("❌ 라벨 분포가 심각하게 불균형함")
    elif max_count / min_count > 10:
        print("⚠️ 라벨 분포가 불균형함")
    else:
        print("✅ 라벨 분포가 비교적 균등함")
    
    # 스코어 문제
    if scores.mean() < 0.1:
        print("❌ 평균 스코어가 너무 낮음")
    elif scores.mean() < 0.3:
        print("⚠️ 평균 스코어가 낮음")
    else:
        print("✅ 평균 스코어가 적절함")
    
    # Segment 길이 문제
    if segment_lengths.mean() < 10:
        print("❌ Segment 길이가 너무 짧음")
    elif segment_lengths.mean() > 500:
        print("❌ Segment 길이가 너무 김")
    else:
        print("✅ Segment 길이가 적절함")
    
    # 5. 샘플 예측 확인
    print("\n=== 5. 샘플 예측 (처음 5개) ===")
    first_video = list(results['results'].keys())[0]
    for i, pred in enumerate(results['results'][first_video][:5]):
        class_name = class_names[pred['label']] if pred['label'] < len(class_names) else f"Unknown_{pred['label']}"
        print(f"  예측 {i+1}: segment={pred['segment']}, label={pred['label']}({class_name}), score={pred['score']:.4f}")

if __name__ == "__main__":
    analyze_prediction_distributions() 