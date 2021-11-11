import glob
import shutil
import os

# 数据集目录
path = "./dataset_kaggledogvscat"
# 训练集目录
train_path = path + '/train'
# 测试集目录
val_path = path + '/val'


# 将某类图片移动到该类的文件夹下
def img_to_file(path):
    print("=========开始移动图片============")
    # 如果没有dog类和cat类文件夹，则新建
    if not os.path.exists(path + "/dog"):
        os.makedirs(path + "/dog")
    if not os.path.exists(path + "/cat"):
        os.makedirs(path + "/cat")
    print("共：{}张图片".format(len(glob.glob(path + "/*.jpg"))))
    # 通过glob遍历到所有的.jpg文件
    for imgPath in glob.glob(path + "/*.jpg"):
        # print(imgPath)
        # 使用/划分
        img = imgPath.strip("\n").replace("\\", "/").split("/")
        # print(img)
        # 将图片移动到指定的文件夹中
        if img[-1].split(".")[0] == "cat":
            shutil.move(imgPath, path + "/cat")
        if img[-1].split(".")[0] == "dog":
            shutil.move(imgPath, path + "/dog")
    print("=========移动图片完成============")


img_to_file(train_path)
print("训练集猫共：{}张图片".format(len(glob.glob(train_path + "/cat/*.jpg"))))
print("训练集狗共：{}张图片".format(len(glob.glob(train_path + "/dog/*.jpg"))))

import random

def split_train_test(fileDir, tarDir):
    if not os.path.exists(tarDir):
        os.makedirs(tarDir)
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print("=========开始移动图片============")
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
    print("=========移动图片完成============")


split_train_test(train_path + '/dog/', val_path + '/dog/')
split_train_test(train_path + '/cat/', val_path + '/cat/')
