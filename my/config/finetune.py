# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=18,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset settings
dataset_type = 'ImageNet'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=448),
    # dict(type='RandomResizedCrop', size=448),
    dict(type="Rotate",angle=90.0,interpolation='bilinear',prob=0.8),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=448),
    # dict(type='Resize', size=(224, -1)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data_root_dir = "/home/jyc/arashi/PycharmProjects/mmclassification/data/"
classes_dir= data_root_dir + "imagenet/classes.txt",
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=data_root_dir+'imagenet/trainval',
        classes= classes_dir,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_root_dir+'imagenet/test',
        classes= classes_dir,
        ann_file=data_root_dir+'imagenet/test.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_root_dir+'imagenet/test',
        classes= classes_dir,
        ann_file=data_root_dir+'imagenet/test.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=["loss",'accuracy'])

# optimizer
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam',lr=0.004)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10,15,19])
runner = dict(type='EpochBasedRunner', max_epochs=20)



# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/home/jyc/arashi/PycharmProjects/mmclassification/checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = "../work_dirs/finetune"
