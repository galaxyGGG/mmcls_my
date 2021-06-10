import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import os

def plot_hist(x,title="title",xlabel="x",ylabel="y"):
    plt.figure() #初始化一张图
    plt.hist(x)  #直方图关键操作
    plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

data_dir = "/home/jyc/arashi/PycharmProjects/mmclassification/data/imagenet/train"
width=[]
height=[]

class_dirs = os.listdir(data_dir)
for cls in class_dirs:
    files = os.listdir(os.path.join(data_dir, cls))
    for file_name in files:
        file_path = os.path.join(data_dir, cls, file_name)
        im=cv2.imread(file_path)
        height.append(im.shape[0])
        width.append(im.shape[1])

plot_hist(width,title="width")
plot_hist(height,title="height")