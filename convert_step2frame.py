import json, argparse, pathlib

def convert_segments(obj, stride):
    if isinstance(obj, dict):
        if "segment" in obj and isinstance(obj["segment"], (list, tuple)):
            s, e = obj["segment"]
            obj["segment"] = [s * stride, e * stride]
        for v in obj.values():
            convert_segments(v, stride)
    elif isinstance(obj, list):
        for v in obj:
            convert_segments(v, stride)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="step 단위 예측 JSON")
    ap.add_argument("--dst", required=True, help="프레임 단위 저장 JSON")
    ap.add_argument("--feature_stride", type=int, default=4,
                    help="1 step = 몇 frame (VideoMAE-S 기본은 4)")
    args = ap.parse_args()

    data = json.load(open(args.src, encoding="utf-8"))
    convert_segments(data, args.feature_stride)

    pathlib.Path(args.dst).write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    print(f"✔ 변환 완료 → {args.dst}")

if __name__ == "__main__":
    main()