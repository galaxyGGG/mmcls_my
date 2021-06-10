_base_ = [
    '../../configs/_base_/models/resnet50.py', '../../configs/_base_/datasets/imagenet_bs32.py',
    '../../configs/_base_/schedules/imagenet_bs256.py', '../../configs/_base_/default_runtime.py'
]

# model settings
model = dict(
    head=dict(
        num_classes=2,
        topk=(1, )
    ))

classes_txt = "/home/jyc/arashi/data/dogs-vs-cats/classes.txt"
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        data_prefix='/home/jyc/arashi/data/dogs-vs-cats/train',
        classes=classes_txt
    ),
    val=dict(
        data_prefix='/home/jyc/arashi/data/dogs-vs-cats/val',
        ann_file='/home/jyc/arashi/data/dogs-vs-cats/val.txt',
        classes=classes_txt
    ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        data_prefix='/home/jyc/arashi/data/dogs-vs-cats//val',
        ann_file='/home/jyc/arashi/data/dogs-vs-cats/val.txt',
        classes=classes_txt
    ))
evaluation = dict(metric_options={"topk":(1,)})

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=10)


load_from = "/home/jyc/arashi/PycharmProjects/mmclassification/checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth"
work_dir = "../work_dirs/cats_vs_dogs"
