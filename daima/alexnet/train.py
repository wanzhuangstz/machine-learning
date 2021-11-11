import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
from datetime import datetime as dt
from Plot import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪24*224像素大小
                                 transforms.RandomHorizontalFlip(),  # 水平随即反转
                                 transforms.ToTensor(),  # 转化成tensor
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = os.path.abspath(os.path.join(os.getcwd(), ""))
# get data root path#获取数据集的根目录（os.getcwd()：获取当前稳健所在目录；os.path.join合并到一起）
image_path = data_root + "/"  # flower data set path

train_dataset = datasets.ImageFolder(root=image_path + "/train",  # 加载数据集，train下面每一类是一个文件夹
                                     transform=data_transform["train"])  # transform是数据预处理（之前定义的），map
train_num = len(train_dataset)  # 数据集有多少张图

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)

cd_list = train_dataset.class_to_idx  # 获取类的名称对应的索引
cla_dict = dict((val, key) for key, val in cd_list.items())  # 键值反转
# write dict into json file
json_str = json.dumps(cla_dict, indent=2)  # 编码成json的格
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=2)

validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=32, shuffle=True,
                                              num_workers=2)



net = AlexNet(num_classes=2, init_weights=True)  # 实例化，分类集有2类，初始化权重是true

# model_weight_path = "./AlexNet.pth"
# net.load_state_dict(torch.load(model_weight_path))

net.to(device)
# 损失函数与优化器
loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './AlexNet.pth'
best_acc = 0.0  # 用来保存最佳平均准确率，为了保存效果最好的一次模型
t1 = dt.now().replace(microsecond=0)
epoch = 50
for i in range(epoch):  # 50轮
    # train
    net.train()  # 用net.train()与net.eavl() 因为用了dropout，希望只在训练时失活，所以用这个来管理dropout
    running_loss = 0.0
    t2 = dt.now().replace(microsecond=0)  # 统计训练一个epoch所使用的时间
    for step, data in enumerate(train_loader, start=0):  # 遍历数据
        images, labels = data  # 将数据分成图像与对应的标签
        optimizer.zero_grad()  # 清空梯度信息
        outputs = net(images.to(device))  # 正向传播，将图像也指认到设备�?
        loss = loss_function(outputs, labels.to(device))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        # print statistics
        running_loss += loss.item()  # 将loss的值累加到runningloss中（loss.item才是loss的值）
        # print train process，打印训练进程
        rate = (step + 1) / len(train_loader)
        if (step+1) % 22 == 0:
            print("\r已运行训练集的{:.1f}%，train loss: {:.3f}".format((rate * 100), loss), end="")
    print()

    # validate验证
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():  # 禁止损失跟踪
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            # max(outputs, dim=1)[1]表示返回最大值的索引，即返回0或者1，
            # 正好与我们原始数据的label相对应,dim=1表示按行找
            acc += (predict_y == val_labels.to(device)).sum().item()  # 计算准确个数
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)  # 保存权重
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (i + 1, running_loss / step, val_accurate))
    print(f'第{i + 1}轮花费时间为: {dt.now().replace(microsecond=0) - t2}')
    plot_save(running_loss / step, val_accurate)

print('Finished Training')
print('模型识最好精确率为%.1f%%' % (best_acc * 100))
print(f'总共花费时间为: {dt.now().replace(microsecond=0) - t1}')
polt_loss(epoch) # 画图
