# Inherit a base config for our model
_base_ = ['../../../../../../../opt/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']

# change the number of classes of the model
model = dict(
    type='FasterRCNN',
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(
            num_classes=10)
    )
)


# Modify dataset related settings
dataset_type = 'CocoDataset'
data_root = '../../../../../../data/V3/'


classes = ('cigarette', 'knife', 'Automatic Rifle', 'Bazooka', 'Grenade Launcher', 'Handgun', 'shotgun', 'SMG', 'Sniper', 'Sword')


# pipelines
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 360), keep_ratio=True),s
    dict(type='PackDetInputs') # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
]
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 360), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 360), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='coco/train/train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline
        )
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='coco/val/val.json',
        data_prefix=dict(img='images/val'),
        pipeline=val_pipeline
        )
    )

val_evaluator = dict(
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + 'coco/val/val.json',
    metric=['bbox'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False,
    backend_args=backend_args)

# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        metainfo=dict(classes=classes),
        type=dataset_type,
        data_root=data_root,
        ann_file='coco/test/test.json',
        data_prefix=dict(img='images/test'),
        pipeline=test_pipeline))


test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/test/test.json',
    metric=['bbox'],  # Metrics to be evaluated
    format_only=True,  
    outfile_prefix='./work_dirs/coco_detection/test')  # The prefix of output json files



# setting optimizer
optim_wrapper = dict(  # Optimizer wrapper config
    type='OptimWrapper',  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
    optimizer=dict(  # Optimizer config. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # Stochastic gradient descent optimizer
        lr=0.02,  # The base learning rate
        momentum=0.9,  # Stochastic gradient descent with momentum
        weight_decay=0.0001),  # Weight decay of SGD
    clip_grad=None,  # Gradient clip option. Set None to disable gradient clip. Find usage in https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
)

param_scheduler = [
    # Linear learning rate warm-up scheduler
    dict(
        type='LinearLR',  # Use linear policy to warmup learning rate
        start_factor=0.001, # The ratio of the starting learning rate used for warmup
        by_epoch=False,  # The warmup learning rate is updated by iteration
        begin=0,  # Start from the first iteration
        end=500),  # End the warmup at the 500th iteration
    # The main LRScheduler
    dict(
        type='MultiStepLR',  # Use multi-step learning rate policy during training
        by_epoch=True,  # The learning rate is updated by epoch
        begin=0,   # Start from the first epoch
        end=12,  # End at the 12th epoch
        milestones=[8, 11],  # Epochs to decay the learning rate
        gamma=0.1)  # The learning rate decay ratio
]

# setting hook
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
    logger=dict(type='LoggerHook', interval=50),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
    param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters of optimizer
    checkpoint=dict(type='CheckpointHook', save_best="coco/bbox_mAP"), # Save checkpoints periodically
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Ensure distributed Sampler shuffle is active
    visualization=dict(type='DetVisualizationHook'))  # Detection Visualization Hook. Used to visualize validation and testing process prediction results


vis_backends = [
    dict(type='LocalVisBackend', save_dir='work_dirs',),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# settiing logger
log_processor = dict(
    type='LogProcessor',  # Log processor to process runtime logs
    window_size=50,  # Smooth interval of log values
    by_epoch=True)  # Whether to format logs with epoch type. Should be consistent with the train loop's type.


log_level = 'INFO'  # The level of logging.
# use a pre-trained model
# Load model checkpoint as a pre-trained model from a given path. This will not resume training.
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dirs`.

# setting trian test cfg
train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=100,  # Maximum training epochs
    val_interval=1)  # Validation intervals. Run validation every epoch.
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type