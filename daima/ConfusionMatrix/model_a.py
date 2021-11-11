import torch.nn as nn
import torch


print(torch.cuda.is_available())


class AlexNet(nn.Module):  # 继承model类
    def __init__(self, num_classes=1000, init_weights=False):  # 初始化参数，定义参数与层结构
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(  # 分类器
            nn.Dropout(p=0.5),  # dropout的方法上全连接层随机失活（一般放在全裂阶层之间）p值随即失火的比例
            nn.Linear(128 * 6 * 6, 2048),  # linear是全连接层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),  # 输出是数据集的类别个数
        )
        if init_weights:  # 初始化权重，定义在下面
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):  # 其实不用，目前pytorch自动就是这个
        for m in self.modules():  # 会返回一个迭代器，遍历模型中所有的模块（遍历每一个层结构）
            if isinstance(m, nn.Conv2d):  # 是否是卷积
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 是就去kaiming_normal初始化
                if m.bias is not None:  # 偏置不是0就置0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
