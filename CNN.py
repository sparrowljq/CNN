# torch.nn是专门为神经网络设计的模块化接口。nn构建于Autograd之上，可用来定义和运行神经网络
# 该案例是一个基础的前向传播（feed-forward）
# 网络：接收输入，经过层层传递运算，得到输出
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as  optiom
class Net(nn.Module):
    # 把网络中把可学习的参数的层放在构造函数__init__中
    def __init__(self):
        # nn.Module子类的构造函数中必须执行父类的构造函数
        super(Net, self).__init__()
        # 卷积层 1表示单通道，6表示通道数，5表示卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5);
        # 全连接层 y =Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        #卷积->激活->池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape,'-1'表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net =Net()
# print(net)
# 网络的可学习参数可以通过net.parameters()返回，net.named_parameters可同时返回可学习的参数及名称
# params = list(net.parameters())
# 可学习参数列表长度为10
# print(len(params))
# 遍历可学习参数
# print(net.named_parameters())
# 遍历每个可学习参数
# for name, param in net.named_parameters():
#      print(name, param.size())

# conv2d的输入必须是四维的，即样本数*通道数*高*宽
input = Variable(t.randn(1, 1, 32, 32))
# out = net(input)
# print(out.size())
# 将所有参数的梯度清零
# net.zero_grad()
# 反向传播
# out.backward(Variable(t.ones(1, 10)))
# out = net(input)
# target = Variable(t.arange(0., 10.))
criterion = nn.MSELoss()
# print(out)
# print(target)
# loss = criterion(out, target)
# print(loss)
# print(loss.grad_fn)
# net.zero_grad()
# 反向传播之前conv1.bias的梯度
# 输出结果为tensor([0., 0., 0., 0., 0., 0.])
# print(net.conv1.bias.grad)
# loss.backward()
# print(net.conv1.bias.grad)

# 在反向传播计算完所有参数的梯度后，还需要使用优化方法更新网络的权重和参数
# 使用随机梯度下降法（SGD）的更新策略如下:
# weight = weight - learning_rate * gradient
# lr =0.01 表示学习率
target = Variable(t.arange(0., 10.))
optimizer = optiom.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()

out = net(input)
loss = criterion(out, target)
loss.backward()
# 更新参数
optimizer.step()

# 数据加载与预处理
# pytorch提供了一些简化和加快数据流程的工具。同时，对于常用的数据集，
# Pytorch也提供了封装好的接口供用户快速的调用，这些数据集主要保存在torchvision中。
# torchvision实现了常用的图像数据加载功能





