import json
import os

def fix_prediction_labels():
    """예측 결과의 라벨을 문자열에서 정수 인덱스로 변환"""
    
    # 클래스 맵 로드
    class_map_file = "data/PKU-MMD/class_map.txt"
    with open(class_map_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    class_map = {name: i for i, name in enumerate(class_names)}
    print(f"클래스 맵 로드 완료: {len(class_map)}개 클래스")
    
    # 결과 파일 로드
    result_file = "work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0/result_detection.json"
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    print("라벨 변환 시작...")
    fixed_count = 0
    error_count = 0
    
    # 각 비디오의 예측 결과 처리
    for video_id, predictions in results['results'].items():
        for pred in predictions:
            label = pred['label']
            
            if isinstance(label, str):
                # 문자열 라벨을 정수 인덱스로 변환
                if label in class_map:
                    pred['label'] = class_map[label]
                    fixed_count += 1
                else:
                    print(f"경고: 알 수 없는 라벨 '{label}' 발견")
                    error_count += 1
            elif isinstance(label, int):
                # 이미 정수인 경우 그대로 유지
                pass
            else:
                print(f"경고: 예상치 못한 라벨 타입: {type(label)}, 값: {label}")
                error_count += 1
    
    # 수정된 결과 저장
    fixed_file = "work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0/result_detection_fixed.json"
    with open(fixed_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"변환 완료!")
    print(f"- 수정된 예측: {fixed_count}개")
    print(f"- 오류: {error_count}개")
    print(f"- 저장 위치: {fixed_file}")
    
    # 빠른 검증
    print("\n=== 검증 ===")
    with open(fixed_file, 'r') as f:
        fixed_results = json.load(f)
    
    first_video = list(fixed_results['results'].keys())[0]
    first_pred = fixed_results['results'][first_video][0]
    
    print(f"첫 번째 예측: {first_pred}")
    print(f"라벨 타입: {type(first_pred['label'])}, 값: {first_pred['label']}")
    
    if isinstance(first_pred['label'], int):
        print("✅ 라벨이 정수 인덱스로 변환됨")
    else:
        print("❌ 라벨 변환 실패")

if __name__ == "__main__":
    fix_prediction_labels() 