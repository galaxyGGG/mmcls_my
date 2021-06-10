from mmcls.apis import inference_model, init_model, show_result_pyplot

config_path = "/home/jyc/arashi/PycharmProjects/mmclassification/configs/resnet/resnet50_b32x8_imagenet.py"
checkpoint_path = "/home/jyc/arashi/PycharmProjects/mmclassification/checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth"
img = "/home/jyc/arashi/PycharmProjects/mmclassification/demo/demo.JPEG"
# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device="cuda:0")
# test a single image
result = inference_model(model, img)
# show the results
show_result_pyplot(model, img, result)