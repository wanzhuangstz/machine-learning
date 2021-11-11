import numpy as np
import torch

from matplotlib import pyplot as plt

Loss_list = []
Accuracy_list = []


def plot_save(loss, acc):
    Loss_list.append(loss)
    Accuracy_list.append(acc)


def polt_loss(n):
    acc_list = torch.tensor(Accuracy_list, device='cpu')
    x = np.arange(0, n)
    y1 = acc_list
    y2 = Loss_list
    my_y_ticks1 = np.arange(0.5, 1, 0.05)
    my_y_ticks2 = np.arange(0, 1, 0.1)
    my_x_ticks = np.arange(0, n, 2)

    plt.subplot(2, 1, 1)  # 第一个代表行数，第二个代表列数，第三个代表索引位置
    plt.plot(x, y1)
    plt.title('accuracy and loss')
    plt.ylabel('val accuracy')
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks1)

    plt.subplot(2, 1, 2)
    plt.plot(x, y2)
    plt.xlabel('epoches')
    plt.ylabel('All_avg_loss')
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks2)

    plt.savefig("./acc_loss.jpg")
    plt.show()
