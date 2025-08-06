import os
import numpy as np
from copy import deepcopy
from .builder import DATASETS
from mmengine.registry import TRANSFORMS
from mmengine.dataset import Compose

@DATASETS.register_module()
class PKUMMDLoader(object):
    def __init__(self, video_dir, anno_dir, split_file, actions_file, split='train', pipeline=None, test_mode=False, **kwargs):
        # PKU-MMD 전용 설정
        self.video_dir = video_dir
        self.anno_dir = anno_dir
        self.split_file = split_file
        self.actions_file = actions_file
        self.split = split
        self.class_names = self._load_class_names()
        self.class_map = {name: i for i, name in enumerate(self.class_names)}
        
        # 기본 속성들 설정
        self.test_mode = test_mode
        self.sample_stride = kwargs.get('sample_stride', 1)
        self.snippet_stride = kwargs.get('snippet_stride', 1)
        self.offset_frames = kwargs.get('offset_frames', 0)
        self.data_path = video_dir
        self.window_size = kwargs.get('window_size', 512)
        self.fps = kwargs.get('fps', 30)  # PKU-MMD FPS 설정을 30으로 변경
        
        # SlidingWindowDataset 파라미터들
        self.feature_stride = kwargs.get('feature_stride', 1)
        self.window_overlap_ratio = kwargs.get('window_overlap_ratio', 0.25)
        self.ioa_thresh = kwargs.get('ioa_thresh', 0.75)
        self.window_stride = int(self.window_size * (1 - self.window_overlap_ratio))
        
        # pipeline을 Compose로 감싸기
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
        
        self.video_list = self._load_split()
        self.data_list = self._gather_samples()

    def split_video_to_windows(self, video_name, video_info, video_anno):
        """SlidingWindowDataset의 split_video_to_windows 메서드를 PKU-MMD에 맞게 수정"""
        # PKU-MMD는 프레임 단위이므로 fps로 나누어 실제 프레임 수 계산
        if self.fps > 0:
            num_frames = int(video_info["duration"] * self.fps)
        else:
            num_frames = video_info["frame"]

        video_snippet_centers = np.arange(0, num_frames, self.snippet_stride)
        snippet_num = len(video_snippet_centers)

        data_list = []
        last_window = False

        for idx in range(max(1, snippet_num // self.window_stride)):
            window_start = idx * self.window_stride
            window_end = window_start + self.window_size

            if window_end > snippet_num:
                window_end = snippet_num
                window_start = max(0, window_end - self.window_size)
                last_window = True

            window_snippet_centers = video_snippet_centers[window_start:window_end]
            window_start_frame = window_snippet_centers[0]
            window_end_frame = window_snippet_centers[-1]

            if (video_anno != {}) and (self.ioa_thresh > 0):
                gt_segments = video_anno["gt_segments"]
                gt_labels = video_anno["gt_labels"]
                anchor = np.array([window_start_frame, window_end_frame])

                # truncate the gt segments inside the window and compute the completeness
                gt_completeness, truncated_gt = self.compute_gt_completeness(gt_segments, anchor)
                valid_idx = gt_completeness > self.ioa_thresh

                # only append window who has gt
                if np.sum(valid_idx) > 0:
                    window_anno = dict(
                        gt_segments=truncated_gt[valid_idx],
                        gt_labels=gt_labels[valid_idx],
                    )
                    data_list.append(
                        [
                            video_name,
                            video_info,
                            window_anno,
                            window_snippet_centers,
                        ]
                    )
            else:
                data_list.append(
                    [
                        video_name,
                        video_info,
                        video_anno,
                        window_snippet_centers,
                    ]
                )

            if last_window:
                break

        return data_list

    def compute_gt_completeness(self, gt_boxes, anchors):
        """Compute the completeness of the gt_bboxes."""
        scores = np.zeros(gt_boxes.shape[0])
        valid_idx = np.logical_and(gt_boxes[:, 0] < anchors[1], gt_boxes[:, 1] > anchors[0])
        valid_gt_boxes = gt_boxes[valid_idx]

        truncated_valid_gt_len = np.minimum(valid_gt_boxes[:, 1], anchors[1]) - np.maximum(valid_gt_boxes[:, 0], anchors[0])
        original_gt_len = valid_gt_boxes[:, 1] - valid_gt_boxes[:, 0]
        scores[valid_idx] = truncated_valid_gt_len / original_gt_len

        truncated_gt = np.copy(gt_boxes)
        truncated_gt[:, 0] = np.maximum(truncated_gt[:, 0], anchors[0])
        truncated_gt[:, 1] = np.minimum(truncated_gt[:, 1], anchors[1])

        return scores, truncated_gt

    def _load_class_names(self):
        import pandas as pd
        try:
            df = pd.read_excel(self.actions_file)
            print(f"엑셀 파일 로드 성공: {self.actions_file}")
            print(f"컬럼명: {df.columns.tolist()}")
            print(f"총 행 수: {len(df)}")
            
            if 'Action' in df.columns:
                class_names = [str(x).strip() for x in df['Action'] if pd.notna(x) and str(x).strip()]
                print(f"액션 클래스 개수: {len(class_names)}")
                print(f"처음 5개 액션: {class_names[:5]}")
                return class_names
            else:
                print(f"Error: 'Action' 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {df.columns.tolist()}")
                return []
        except Exception as e:
            print(f"Error: 엑셀 파일 로드 실패 - {self.actions_file}")
            print(f"Error details: {e}")
            return []

    def _load_split(self):
        with open(self.split_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # PKU-MMD split 파일의 실제 키 확인
        if self.split == 'train':
            key = 'Training videos:'
        else:  # test
            # 실제 파일에서 사용되는 키 확인
            possible_keys = ['Test videos:', 'Validation videos:', 'Testing videos:']
            key = None
            for k in possible_keys:
                if any(k in l for l in lines):
                    key = k
                    break
            if key is None:
                # 기본값으로 설정
                key = 'Test videos:'
        
        idx = [i for i, l in enumerate(lines) if key in l]
        if not idx:
            print(f"Warning: '{key}' 키를 찾을 수 없습니다. 전체 파일 내용:")
            for i, l in enumerate(lines[:10]):
                print(f"Line {i}: {l.strip()}")
            return []
        
        idx = idx[0]
        vids = []
        for l in lines[idx+1:]:
            if ':' in l: break
            vids += [v.strip() for v in l.strip().split(',') if v.strip()]
        return vids

    def _gather_samples(self):
        samples = []
        for vid in self.video_list:
            video_path = os.path.join(self.video_dir, f"{vid}.avi")
            anno_path = os.path.join(self.anno_dir, f"{vid}.txt")
            if not os.path.exists(video_path) or not os.path.exists(anno_path):
                continue

            # 비디오 정보 구성 (thumos 형식에 맞춤)
            video_info = {
                "video_name": vid,
                "frame": 0,  # 실제 프레임 수는 나중에 계산
                "duration": 0,  # 실제 duration은 나중에 계산
                "annotations": []
            }

            max_end_frame = 0  # 가장 큰 end_frame 찾기

            # 어노테이션 로드
            with open(anno_path, 'r', encoding='utf-8') as f:
                for line in f:
                    arr = line.strip().split(',')
                    if len(arr) < 4: continue
                    label_idx = int(arr[0])
                    start_frame = int(arr[1])
                    end_frame = int(arr[2])
                    conf = float(arr[3])

                    # 가장 큰 end_frame 업데이트
                    max_end_frame = max(max_end_frame, end_frame)

                    # class_names에서 실제 라벨 이름 가져오기
                    if label_idx < len(self.class_names):
                        label_name = self.class_names[label_idx]
                        video_info["annotations"].append({
                            "label": label_name,
                            "segment": [start_frame, end_frame],
                            "confidence": conf
                        })

            # duration 계산 (프레임을 초 단위로 변환, PKU-MMD FPS 기준)
            if max_end_frame > 0:
                video_info["duration"] = max_end_frame / 30.0  # PKU-MMD는 30fps로 변경
                video_info["frame"] = max_end_frame

            # GT 어노테이션 생성
            video_anno = self.get_gt(video_info)
            if video_anno != {}:
                # SlidingWindowDataset의 split_video_to_windows 메서드 사용
                window_samples = self.split_video_to_windows(vid, video_info, video_anno)
                samples.extend(window_samples)

        return samples

    def get_gt(self, video_info, thresh=0.0):
        """thumos.py의 get_gt 메서드와 유사한 구조로 구현"""
        gt_segment = []
        gt_label = []

        for anno in video_info["annotations"]:
            # PKU-MMD는 이미 프레임 단위로 되어있으므로 변환 불필요
            gt_start = anno["segment"][0]
            gt_end = anno["segment"][1]

            # 최소 길이 필터링
            if gt_end - gt_start > thresh:
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map[anno["label"]])

        if len(gt_segment) == 0:  # 유효한 GT가 없음
            return {}
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return annotation

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # 원본 수정 방지
            # SlidingWindowDataset과 동일한 방식으로 window 시작점 적용
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                window_size=self.window_size,
                # trunc window setting
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                # sliding post process setting
                fps=30,  # PKU-MMD는 30fps로 변경
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],  # 올바른 window 시작점
                duration=video_info.get("duration", 0),
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
        return results

    def __len__(self):
        return len(self.data_list)