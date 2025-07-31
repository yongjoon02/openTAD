#!/usr/bin/env python3
"""
scale_check_json.py  ―  PKU-MMD GT vs AdaTAD E2E 예측 스케일 확인

사용 예)
  python scale_check_json.py \
      --gt "F:/OpenTAD/data/PKU-MMD/annotations_test.json" \
      --pred "F:/OpenTAD/work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0/result_detection.json" \
      --feature_stride 4
"""
import json, argparse, numpy as np
from collections import defaultdict

# ---------- Loader ----------
def load_gt(json_path):
    from collections import defaultdict
    data = json.load(open(json_path, encoding="utf-8"))
    gt = defaultdict(list)

    def _add(vid, seg):
        if seg and len(seg) == 2:
            gt[vid].append(tuple(seg))

    if isinstance(data, dict):          # {video_id: meta}
        for vid, meta in data.items():
            for ann in meta.get("annotations", []):
                _add(vid, ann.get("segment") or ann.get("frame"))
    elif isinstance(data, list):        # [ {...}, {...} ]
        for item in data:
            vid = item.get("video_id") or item.get("video") or item.get("video_name")
            if "annotations" in item:   # 상위에 duration 등 포함
                for ann in item["annotations"]:
                    _add(vid, ann.get("segment") or ann.get("frame"))
            else:                       # 단일 주석 행
                _add(vid, item.get("segment") or item.get("frame"))
    else:
        raise ValueError("Unsupported GT JSON format")

    return gt


from collections import defaultdict
import json

def load_pred(json_path):
    """video_id ➜ [ {segment:[s,e], ...}, … ]  형식 전용"""
    raw = json.load(open(json_path, encoding="utf-8"))
    pred = defaultdict(list)
    
    # results 키가 있는지 확인
    if "results" in raw:
        raw = raw["results"]
    
    for vid, dets in raw.items():
        for det in dets:
            if isinstance(det, dict) and "segment" in det:
                s, e = det["segment"]
                pred[vid].append((float(s), float(e)))
            else:
                print(f"Warning: Unexpected prediction format for video {vid}: {det}")
    
    return pred



# ---------- 비교 ----------
def compare(gt, pred, stride):
    len_diff, start_diff = [], []
    matched_videos = 0
    
    print(f"GT 비디오 수: {len(gt)}")
    print(f"예측 비디오 수: {len(pred)}")
    
    for vid in pred:
        if vid not in gt:
            print(f"Warning: 예측에만 있는 비디오: {vid}")
            continue
        matched_videos += 1
        
        # 예측과 GT의 개수가 다를 수 있으므로 더 짧은 쪽에 맞춤
        min_len = min(len(pred[vid]), len(gt[vid]))
        for i in range(min_len):
            ps, pe = pred[vid][i]
            gs, ge = gt[vid][i]
            ps_f, pe_f = ps * stride, pe * stride   # 특징 → 프레임
            len_diff.append((pe_f - ps_f) - (ge - gs))
            start_diff.append(ps_f - gs)
    
    print(f"매칭된 비디오 수: {matched_videos}")
    print(f"총 비교 세그먼트 수: {len(len_diff)}")
    
    if len(len_diff) == 0:
        print("Warning: 매칭되는 세그먼트가 없습니다!")
        return np.array([]), np.array([])
    
    return np.array(len_diff), np.array(start_diff)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",   required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--feature_stride", type=int, required=True,
                    help="특징 1 step 이 몇 프레임인지 (백본 downsample factor)")
    args = ap.parse_args()

    gt   = load_gt(args.gt)
    pred = load_pred(args.pred)

    len_d, st_d = compare(gt, pred, args.feature_stride)
    print(f"\n=== Scale Check  (feature_stride = {args.feature_stride}) ===")
    
    if len(len_d) == 0:
        print("❌ 비교할 데이터가 없습니다!")
        return
    
    for name, diff in [("Length Δ", len_d), ("Start Δ", st_d)]:
        print(f"{name:9}:  mean={diff.mean():6.2f}  median={np.median(diff):6.2f}  "
              f"std={diff.std():6.2f}  min={diff.min():6.1f}  max={diff.max():6.1f}")
    print("-------------------------------------------------------------")
    print("※ |mean|·|median| 이 0~2 프레임 내에 있고, std 가 작으면 스케일이 맞습니다.\n")

if __name__ == "__main__":
    main()
