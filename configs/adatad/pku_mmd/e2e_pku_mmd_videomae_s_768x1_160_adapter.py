
_base_ = [
    "../../_base_/datasets/pku-mmd/pku-mmd.py",  
    "../../_base_/models/actionformer.py",
]


scale_factor = 1  
chunk_num = 512 * scale_factor // 16  


dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=4), 
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=512,  
                trunc_thresh=0.5,
                crop_ratio=[0.9, 1.0],
                scale_factor=1,  
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=4),  
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1), 
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160), 
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=4), 
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),  
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

model = dict(
   
    rpn_head = dict(
    num_classes=51,
    prior_generator=dict(
        type="PointGenerator",
        strides=[1, 2, 4, 8, 16, 32],  
        regression_range=[                             
            (0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000) 
        ],
    ),
    center_sample_radius=1.5,
    ),
    backbone=dict(
        type="VisionTransformerAdapter",
        img_size=160,
        patch_size=16,
        embed_dims=384,  
        depth=12,
        num_heads=6, 
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        num_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        pretrained="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
        return_feat_map=True,
        with_cp=True,
        adapter_mlp_ratio=0.25,
        total_frames=512,  # 576에서 512로 되돌림
        adapter_index=list(range(12)),
        custom=dict(
            norm_eval=False,
            freeze_backbone=False,  # 백본 unfreeze로 변경
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t -> b c t", reduction="mean"),
            ],
        ),
    ),
    projection=dict(
        in_channels=384, 
        out_channels=512,   
        max_seq_len=512,  
        attn_cfg=dict(n_mha_win_size=-1),  
    ),
)

solver = dict(
    train=dict(batch_size=8, num_workers=4),   # 배치 크기 16->8, 워커 수 4->2로 감소
    val=dict(batch_size=8, num_workers=4),     # 배치 크기 16->8, 워커 수 4->2로 감소
    test=dict(batch_size=4, num_workers=4),    # 배치 크기 4->2, 워커 수 4->2로 감소
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=1e-5,  # 하위 블록용 낮은 학습률
        weight_decay=0.05,
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.05),  # adapter
            dict(name="blocks.10", lr=5e-5, weight_decay=0.05),  # 상위 블록 2개 (10, 11)
            dict(name="blocks.11", lr=5e-5, weight_decay=0.05),
        ],
        exclude=["backbone"],
    ),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=120)  # warm-up 3에폭으로 증가

inference = dict(
    load_from_raw_predictions=False, 
    save_raw_prediction=False,
    score_thresh=0.05,  
)
post_processing = dict(
    nms=dict(
        use_soft_nms=False,
        sigma=0.3,  
        max_seg_num=1000,  
        multiclass=False, 
        voting_thresh=0.6,  
    ),
    save_dict=True,
)

workflow = dict(
    logging_interval=5,
    checkpoint_interval=3,
    val_loss_interval=3,     
    val_eval_interval=3,      
    val_start_epoch=3,       
    end_epoch=120,           
    num_sanity_check=0,    
)

work_dir = "work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter"

# 완전히 새로 시작 - 체크포인트 로드 없음
# load_from = None
# resume = False

evaluation = dict(
    type="mAP_PKU_MMD",
    subset="validation",  
    tiou_thresholds=[0.2, 0.3, 0.4, 0.5, 0.6],
    ground_truth_filename="data/PKU-MMD/pku_val.json",  
)
