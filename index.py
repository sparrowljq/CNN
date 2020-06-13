# pytorch基础
import torch as t
import numpy as np
from torch.autograd import Variable
# 创建 张量
# x = t.Tensor(5, 3)
# 构建5*3的矩阵，只是分配了空间，未出化
# print(x)
# 使用[0, 1]均匀分布随机初始化二维数组
# x = t.rand(5, 3)
# 输出结果为
# tensor([[0.0881, 0.4624, 0.0111],
#         [0.3290, 0.3680, 0.4934],
#         [0.7264, 0.7859, 0.3529],
#         [0.3885, 0.4358, 0.7851],
#         [0.8414, 0.7236, 0.7904]])
# print(x)
# 查看x的形状
# 输出结果为torch.Size([5, 3])
# print(x.size())

# y = t.rand(5, 3)
# 求两个张量和
# 输出结果为tensor([[1.5460, 0.9761, 0.5159],
#         [1.0799, 0.9972, 0.8201],
#         [1.4384, 0.4026, 1.7021],
#         [1.2841, 0.9324, 1.0867],
#         [0.4005, 0.5356, 0.8422]])
# print(x+y)
# 第二种写法
# print(t.add(x, y))
# 将结果存放到指定的张量中
# result = t.Tensor(5, 3)
# t.add(x, y, out=result)
# print(result)
# 普通加法y的值不变
# y.add(x)
# print(y)
# 第二种加法y的值改变
# y.add_(x)
# 注意：函数名后面带下划线_的函数会修改Tensor本身。如x.add_()
# Tensor的切片操作与numpy类似
# 输出结果为
# tensor([[0.7963, 0.5142, 0.7294],
#         [0.1810, 0.4269, 0.4417],
#         [0.5161, 0.9221, 0.8907],
#         [0.5564, 0.4678, 0.3854],
#         [0.9729, 0.1117, 0.7006]])
# print(x)
# 输出第二列的所有元素
# 输出结果为
# tensor([0.5142, 0.4269, 0.9221, 0.4678, 0.1117])
# print(x[:, 1])
# Tensor与numpy之间的转换
# 将Tensor->numpy
# a = t.ones(5)
# b = a.numpy()
# 输出结果为[1. 1. 1. 1. 1.]
# print(b)
# 将numpy->Tensor
# a = np.ones(5)
# b = t.from_numpy(a)
# 输出结果为tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
# print(b)
# Tensor和numpy共享内存，所以他们之间的转换很快，而且几乎不会消耗资源。
# 这就意味着当其中的一个改变是另一个也会随之改变
# b.add_(1)
# 输出结果为tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
# print(b)
# 输出结果为[2. 2. 2. 2. 2.]
# print(a)

# Tensor可通过cuda方法转为GPU的Tensor
# if t.cuda.is_available():
#     x = x.cuda()
#     y = y.cuda()
#     print(x+y)

# Autograd 自动微分
# DeapLearning本质是通过反向传播求导数，Pytorch的Autograd模块实现了此功能。
# 在Tensor上的所有操作，Autograd都能为他们自动提供微分，避免手动计算导数的复杂过程。
# autograd.Variable是Autograd中的核心类，它简单封装了Tensor,并支持几乎所有的Tensor操作。
# Tensor在被封装了variable之后，可以调用它的.backward实现反向传播，自动计算所有的梯度。
# Variable主要包三个属性：
# data:保存Variable所包含的Tensor
# grad 保存data对应的梯度，grad是个Variable，而不是Tensor,它和data的形状一样
# grad_fn 是指向一个function对象，这个function用来方向传播计算输入的梯度

# 使用Tensor新建一个Variable

# x = Variable(t.ones(2, 2), requires_grad=True)
# print(x)
# y = x.sum()
# 输出结果为tensor(4., grad_fn=<SumBackward0>)
# print(y)
# print(y.grad_fn)
# 反向传播，计算梯度
# print(y.backward())
# 输出结果为tensor([[1., 1.],
#         [1., 1.]])
# print(x.grad)
# grad在反向传播过程中是累计的（accumulated）,这意味着每次运行方向传播，梯度都会累加之前的梯度，
# 所以反向传播之前需要把梯度清零。
# x.grad.data.zero_()
# y.backward()
# 输出结果为tensor([[2., 2.],
#         [2., 2.]])
# 输出结果为tensor([[1., 1.],
#         [1., 1.]])
# print(x.grad)
# 梯度清零操作

# Variable和Tensor具有近乎一致的接口，在实际应用中可以无缝的切换
# x = Variable(t.ones(4, 5))
# y = t.cos(x)
# x_tensor_cos = t.cos(x.data)
# 输出结果为tensor([[0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403]])
# print(y)
# 输出结果为tensor([[0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403],
#         [0.5403, 0.5403, 0.5403, 0.5403, 0.5403]])
# print(x_tensor_cos)



