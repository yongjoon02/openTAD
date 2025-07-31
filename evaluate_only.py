import json
import sys
import os
sys.path.append('.')

from opentad.evaluations import build_evaluator
from opentad.utils import setup_logger

def main():
    # 설정
    work_dir = "work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0"
    result_file = os.path.join(work_dir, "result_detection_frame.json")
    
    # 로거 설정
    logger = setup_logger("Evaluation", save_dir=work_dir, distributed_rank=0)
    
    # 결과 파일 로드
    with open(result_file, 'r') as f:
        result_eval = json.load(f)
    
    # 평가기 설정
    evaluation_cfg = dict(
        type="mAP_PKU_MMD",
        subset="test",
        tiou_thresholds=[0.2, 0.4, 0.5,0.6, 0.7],
        ground_truth_filename="data/PKU-MMD/annotations_test.json",
        prediction_filename=result_eval,
    )
    
    # 평가기 생성 및 실행
    logger.info("Starting evaluation...")
    evaluator = build_evaluator(evaluation_cfg)
    
    # 평가 실행
    metrics_dict = evaluator.evaluate()
    
    # 결과 출력
    evaluator.logging(logger)
    
    # 결과를 JSON으로 저장
    metrics_file = os.path.join(work_dir, "evaluation_results.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {metrics_file}")
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main() 