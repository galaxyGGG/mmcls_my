import random
import os
import shutil

# 数据集目录
path = "/home/jyc/arashi/data/dogs-vs-cats"
# 训练集目录
train_path = path + '/train'
# 测试集目录
test_path = path + '/val'


def split_train_test(fileDir, tarDir):
    if not os.path.exists(tarDir):
        os.makedirs(tarDir)
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    random.seed(0)
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print("=========开始移动图片============")
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
    print("=========移动图片完成============")


# 生成索引文件
def write_img_names_and_cls(dir,classes_path,txt_name="val"):
    class_dirs = os.listdir(dir)
    txt_path = os.path.join(dir, "..", txt_name + ".txt")
    classes=[]
    with open(classes_path,"r") as fc:
        for line in fc.readlines():
            classes.append(line.strip())
    with open(txt_path, "w") as f:
        for cls in class_dirs:
            ind_cls = str(classes.index(cls))
            files = os.listdir(os.path.join(dir,cls))
            for file in files:
                f.write(cls+"/"+file+" "+ind_cls+"\n")


if __name__ == '__main__':
    # split_train_test(train_path + '/dogs/', test_path + '/dogs/')
    # split_train_test(train_path + '/cats/', test_path + '/cats/')
    # write_img_names_and_cls(test_path, path+"/classes.txt", txt_name="val")
    write_img_names_and_cls("/home/jyc/arashi/PycharmProjects/mmclassification/data/imagenet/test", "/home/jyc/arashi/PycharmProjects/mmclassification/data/imagenet/classes.txt", txt_name="test")