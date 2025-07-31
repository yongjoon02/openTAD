import json
import numpy as np
import pandas as pd
import multiprocessing as mp

from .builder import EVALUATORS


@EVALUATORS.register_module()
class mAP_PKU_MMD:
    def __init__(
        self,
        ground_truth_filename,
        prediction_filename,
        subset,
        tiou_thresholds,
        thread=16,
    ):
        super().__init__()

        if not ground_truth_filename:
            raise IOError("Please input a valid ground truth file.")
        if not prediction_filename:
            raise IOError("Please input a valid prediction file.")

        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.thread = thread  # multi-process workers

        # Import ground truth and predictions.
        self.ground_truth = self._import_ground_truth(ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        # Create activity index from ground truth labels (실제 사용되는 클래스만)
        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth["label"].unique()))}
        # Ground truth는 이미 올바른 인덱스이므로 매핑하지 않음
        # Prediction만 매핑 (모델 출력이 0-50 범위라고 가정)
        # 매핑 전에 유효한 라벨만 필터링
        valid_labels = set(self.activity_index.keys())
        self.prediction = self.prediction[self.prediction["label"].isin(valid_labels)]
        if len(self.prediction) > 0:
            self.prediction["label"] = self.prediction["label"].replace(self.activity_index)
        else:
            print("Warning: No valid predictions found after label filtering!")

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file and returns the ground truth instances.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        """
        with open(ground_truth_filename, "r") as fobj:
            data = json.load(fobj)

        # Load class map
        class_map_file = "data/PKU-MMD/class_map.txt"
        with open(class_map_file, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        
        class_index = {name: i for i, name in enumerate(class_names)}

        # Read ground truth data.
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for video_info in data:
            video_name = video_info["video_name"]
            
            # subset 필터링 개선 - JSON 파일에 subset 정보가 없을 수 있음
            # 파일명으로 train/test 구분
            if "train" in self.subset.lower():
                if "test" in ground_truth_filename.lower():
                    continue  # test 파일인데 train 평가하려면 스킵
            else:  # test
                if "train" in ground_truth_filename.lower():
                    continue  # train 파일인데 test 평가하려면 스킵

            for ann in video_info["annotations"]:
                label_name = ann["label"]
                if label_name in class_index:
                    video_lst.append(video_name)
                    t_start_lst.append(float(ann["segment"][0]))  # 프레임 단위
                    t_end_lst.append(float(ann["segment"][1]))    # 프레임 단위
                    label_lst.append(class_index[label_name])

        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        return ground_truth

    def _import_prediction(self, prediction_filename):
        """Reads prediction file and returns the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        # if prediction_filename is a string, then json load
        if isinstance(prediction_filename, str):
            with open(prediction_filename, "r") as fobj:
                data = json.load(fobj)
        elif isinstance(prediction_filename, dict):
            data = prediction_filename
        else:
            raise IOError(f"Type of prediction file is {type(prediction_filename)}.")

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        
        print(f"Prediction data keys: {data.keys()}")
        if "results" in data:
            print(f"Number of videos in results: {len(data['results'])}")
        
        for video_id, v in data["results"].items():
            for result in v:
                try:
                    # 라벨이 문자열인지 숫자인지 확인
                    label = result["label"]
                    if isinstance(label, str):
                        # 문자열 라벨인 경우, 클래스맵에서 인덱스 찾기
                        class_map_file = "data/PKU-MMD/class_map.txt"
                        with open(class_map_file, "r") as f:
                            class_names = [line.strip() for line in f.readlines()]
                        
                        if label in class_names:
                            label_idx = class_names.index(label)
                        else:
                            print(f"Warning: Unknown label '{label}', skipping...")
                            continue
                    else:
                        # 숫자 라벨인 경우
                        label_idx = int(label)
                    
                    # 예측 결과를 프레임 단위로 변환 (feature_stride=4 적용)
                    feature_stride = 4  # VideoMAE-S의 feature_stride
                    t_start_frame = float(result["segment"][0]) * feature_stride
                    t_end_frame = float(result["segment"][1]) * feature_stride
                    
                    video_lst.append(video_id)
                    t_start_lst.append(t_start_frame)
                    t_end_lst.append(t_end_frame)
                    label_lst.append(label_idx)
                    score_lst.append(float(result["score"]))
                    
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Warning: Skipping invalid prediction result: {result}, Error: {e}")
                    continue

        # 배열 길이 확인
        lengths = [len(video_lst), len(t_start_lst), len(t_end_lst), len(label_lst), len(score_lst)]
        print(f"Array lengths: {lengths}")
        
        if len(set(lengths)) != 1:
            print(f"Error: Array lengths are not equal: {lengths}")
            # 가장 짧은 길이에 맞춰 자르기
            min_length = min(lengths)
            video_lst = video_lst[:min_length]
            t_start_lst = t_start_lst[:min_length]
            t_end_lst = t_end_lst[:min_length]
            label_lst = label_lst[:min_length]
            score_lst = score_lst[:min_length]
            print(f"Truncated to length: {min_length}")

        prediction = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "score": score_lst,
            }
        )
        return prediction

    def wrapper_compute_average_precision(self, cidx_list):
        """Computes average precision for a sub class list."""
        for cidx in cidx_list:
            gt_idx = self.ground_truth["label"] == cidx
            pred_idx = self.prediction["label"] == cidx
            self.mAP_result_dict[cidx] = compute_average_precision_detection(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                tiou_thresholds=self.tiou_thresholds,
            )

    def multi_thread_compute_average_precision(self):
        self.mAP_result_dict = mp.Manager().dict()

        num_total = len(self.activity_index.values())
        num_activity_per_thread = num_total // self.thread + 1

        processes = []
        for tid in range(self.thread):
            num_start = int(tid * num_activity_per_thread)
            num_end = min(num_start + num_activity_per_thread, num_total)

            p = mp.Process(
                target=self.wrapper_compute_average_precision,
                args=(list(self.activity_index.values())[num_start:num_end],),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index.items())))
        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = self.mAP_result_dict[i]
        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        # 데이터 검증
        self._validate_data()
        
        self.ap = self.multi_thread_compute_average_precision()
        self.mAPs = self.ap.mean(axis=1)
        self.average_mAP = self.mAPs.mean()

        metric_dict = dict(average_mAP=self.average_mAP)
        for tiou, mAP in zip(self.tiou_thresholds, self.mAPs):
            metric_dict[f"mAP@{tiou}"] = mAP
        return metric_dict

    def _validate_data(self):
        """평가 전 데이터 검증"""
        print(f"=== 데이터 검증 ===")
        print(f"Ground truth 비디오 수: {len(self.ground_truth['video-id'].unique())}")
        print(f"Ground truth 인스턴스 수: {len(self.ground_truth)}")
        print(f"Ground truth 라벨 범위: {self.ground_truth['label'].min()} - {self.ground_truth['label'].max()}")
        print(f"Ground truth 라벨 종류: {sorted(self.ground_truth['label'].unique())}")
        
        print(f"Prediction 비디오 수: {len(self.prediction['video-id'].unique())}")
        print(f"Prediction 인스턴스 수: {len(self.prediction)}")
        print(f"Prediction 라벨 범위: {self.prediction['label'].min()} - {self.prediction['label'].max()}")
        print(f"Prediction 라벨 종류: {sorted(self.prediction['label'].unique())}")
        
        # 공통 비디오 확인
        gt_videos = set(self.ground_truth['video-id'].unique())
        pred_videos = set(self.prediction['video-id'].unique())
        common_videos = gt_videos & pred_videos
        print(f"공통 비디오 수: {len(common_videos)}")
        
        if len(common_videos) == 0:
            raise ValueError("Ground truth와 prediction에 공통 비디오가 없습니다!")
        
        # 라벨 매핑 확인
        print(f"Activity index: {self.activity_index}")
        print(f"=== 검증 완료 ===\n")

    def logging(self, logger=None):
        if logger == None:
            pprint = print
        else:
            pprint = logger.info
        pprint("Evaluating PKU-MMD dataset.")
        pprint("Loaded annotations from {} subset.".format(self.subset))
        pprint("Number of ground truth instances: {}".format(len(self.ground_truth)))
        pprint("Number of predictions: {}".format(len(self.prediction)))
        pprint("Fixed threshold for tiou score: {}".format(self.tiou_thresholds))
        pprint("Average-mAP: {:>4.2f} (%)".format(self.average_mAP * 100))
        for tiou, mAP in zip(self.tiou_thresholds, self.mAPs):
            pprint("mAP at tIoU {:.2f} is {:>4.2f}%".format(tiou, mAP * 100))


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    ap = np.zeros(len(tiou_thresholds))

    for tidx in range(len(tiou_thresholds)):
        # Computing prec-rec
        this_tp = np.cumsum(tp[tidx, :]).astype(float)
        this_fp = np.cumsum(fp[tidx, :]).astype(float)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)
    return ap


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0])
        + (target_segment[1] - target_segment[0])
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union.clip(1e-8)
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap 