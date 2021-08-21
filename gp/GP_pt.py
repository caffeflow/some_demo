import torch.nn.functional as F
from torch import optim
from torchvision import models
from torchvision import transforms
from torchvision import datasets
# import tf.keras.models as models
# import tensorflow as tf
import numpy as np
import os
import torch.nn as nn
import torch

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# batch_size = 256

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device("cpu")
batch_size = 256


# class GradPooling_v2(tf.keras.layers.Layer):
# def __init__(self, pool_size, stride, pad, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
#     super().__init__(trainable, name, dtype, dynamic, **kwargs)
#     self.pool_size = pool_size
#     self.cs = self.pool_size // 2
#     self.stride = stride
#     self.arg_max = None
#     self.pad = pad
class GradPooling_v2(nn.Module):
    def __init__(self, pool_size, stride, pad):
        super(GradPooling_v2, self).__init__()
        self.pool_size = pool_size
        self.cs = self.pool_size // 2
        self.stride = stride
        self.arg_max = None
        self.pad = pad

    # todo
    # def build(self, input_shape):
    #     self.lamb = self.add_weight(name='lamb', shape=[1])
    #     return super().build(input_shape)

    # # todo
    # def compute_diff(self, inputs):
    #     """inputs as nchw's ndarray"""
    #     diff_x_left = inputs[:, :, 2:, 2:] - inputs[:, :, :-2, 2:]
    #     diff_x_right = inputs[:, :, :-2, 2:] - inputs[:, :, 2:, 2:]
    #     diff_y_top = inputs[:, :, 2:, 2:] - inputs[:, :, 2:, :-2]
    #     diff_y_bottom = inputs[:, :, 2:, :-2] - inputs[:, :, 2:, 2:]
    #     diffs = tf.abs(diff_x_right) + tf.abs(diff_y_bottom) + \
    #         tf.abs(diff_x_left) + tf.abs(diff_y_top)
    #     return diffs

    def compute_diff(self, inputs):
        """inputs as nchw's ndarray

            采用梯度算子即可，有其他替代方案，如Sobel。

            另考虑其他思想，如torch.svd
        """
        diff_x_left = inputs[:, :, 2:, 2:] - inputs[:, :, :-2, 2:]
        diff_x_right = inputs[:, :, :-2, 2:] - inputs[:, :, 2:, 2:]
        diff_y_top = inputs[:, :, 2:, 2:] - inputs[:, :, 2:, :-2]
        diff_y_bottom = inputs[:, :, 2:, :-2] - inputs[:, :, 2:, 2:]
        diffs = np.abs(diff_x_right) + np.abs(diff_y_bottom) + \
            np.abs(diff_x_left) + np.abs(diff_y_top)
        return diffs

    # def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
    #     N, C, H, W = input_data.shape
    #     out_h = (H + 2 * pad - filter_h) // stride + 1
    #     out_w = (W + 2 * pad - filter_w) // stride + 1
    #     img = np.pad(input_data, [(0, 0), (0, 0),
    #                               (pad, pad), (pad, pad)], 'constant')

    #     col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    #     diffs = self.compute_diff(img)
    #     diff_col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    #     diffs = np.pad(
    #         diffs, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    #     for y in range(filter_h):
    #         y_max = y + stride * out_h
    #         for x in range(filter_w):
    #             x_max = x + stride * out_w
    #             col[:, :, y, x, :, :] = img[:, :,
    #                                         y:y_max:stride, x:x_max:stride]
    #             diff_col[:, :, y, x, :, :] = diffs[:,
    #                                                :, y:y_max:stride, x:x_max:stride]
    #     col = col.transpose([0, 4, 5, 1, 2, 3]).reshape(N * out_h * out_w, -1)
    #     diff_col = diff_col.transpose(
    #         [0, 4, 5, 1, 2, 3]).reshape(N * out_h * out_w, -1)
    #     return col, diff_col

    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        """inputs as nchw's ndarray, 
            将卷积转换成矩阵乘法   https://zhuanlan.zhihu.com/p/63974249  

            return col:n*oh*ow,c*fh*fw 
            return diff_col:n*oh*ow,c*fh*fw

            # 个人补充：
            输入9x6的矩阵,卷积核3x3，下面将卷积运算转化为矩阵乘法？
            卷积核将遍历输入3x2=6次,每次遍历都对应着输入矩阵上的一个3x3子集
            将上诉子集分别展开为行，记x1、x2、x3、x4、x5、x6
            那么可构建一个矩阵A = [x1，x2，x3，x4，x5，x6].T, A的尺寸为(6,9)
            类似地,将卷积核展开为列,记为k，尺寸为(9,1)
            那么A@k完成矩阵乘法,得到输出y,y的尺寸为(6,1),这里的y(i,1)就是卷积运算一次加权和。
            将y再反着变换形状为”原有形式“, y的最终尺寸为(3,2)，记为卷积运算结果。

            np.lib.stride_tricks.as_strided 可去掉for

            def split_by_strides(X, kh, kw, s):
                # kh，kw 是卷积核的尺寸
                N, C, H, W = X.shape
                oh = (H - kh) // s + 1
                ow = (W - kw) // s + 1
                strides = (*X.strides[:-2], X.strides[-2]*s, X.strides[-1]*s, *X.strides[-2:])
                A = as_strided(X, shape=(N,C,oh,ow,kh,kw), strides=strides)
                return A #（N，C，？，？，kh，kw）
            # 举个的例子
            kn,C,kh, kw = self.filters.shape
            X_split = split_by_strides(X, kh, kw, s)   # X_split.shape: (N, oh, ow, kh, kw, C)
            feature_map = np.tensordot(x, kernel, [(1, 4, 5), (i, 2, 3)]).transpose(0, 3, 1, 2)# tensordot是张量的点乘
            """
        N, C, H, W = input_data.shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1
        # 对nchw四个轴进行padding
        img = np.pad(input_data, [(0, 0), (0, 0),
                                  (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
        diffs = self.compute_diff(img)
        diff_col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
        # nchw上padding
        diffs = np.pad(
            diffs, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :,
                                            y:y_max:stride, x:x_max:stride]
                diff_col[:, :, y, x, :, :] = diffs[:,
                                                   :, y:y_max:stride, x:x_max:stride]
        # n,c,fh,fw,oh,ow -> n,oh,ow,c,fh,fw -> n*oh*ow,c*fh*fw
        col = col.transpose([0, 4, 5, 1, 2, 3]).reshape(N * out_h * out_w, -1)
        diff_col = diff_col.transpose(
            [0, 4, 5, 1, 2, 3]).reshape(N * out_h * out_w, -1)
        return col, diff_col

    # # todo
    # def fc(self, inputs):
    #     inputs = np.transpose(inputs, [0, 3, 1, 2])  # nchw
    #     N, C, H, W = inputs.shape
    #     out_h = (H + 2 * self.pad - self.pool_size) // self.stride + 1
    #     out_w = (W + 2 * self.pad - self.pool_size) // self.stride + 1
    #     col, diff_col = self.im2col(
    #         inputs, self.pool_size, self.pool_size, self.stride, self.pad)
    #     zeros = np.zeros_like(diff_col)
    #     ones = np.ones_like(diff_col)
    #     diff_col = np.where(diff_col > np.mean(diff_col), ones, zeros)
    #     col = col.reshape(-1, self.pool_size * self.pool_size)
    #     diff_col = diff_col.reshape(-1, self.pool_size * self.pool_size)
    #     mid_diff = diff_col[:, [self.pool_size * self.pool_size // 2]]
    #     mid_diff = mid_diff.reshape(-1).astype(np.uint8)
    #     mid_diff = np.eye(2)[mid_diff]
    #     mean_out = np.mean(col, axis=1).reshape(-1)
    #     max_out = np.max(col, axis=1).reshape(-1)
    #     out = np.array(
    #         mean_out * mid_diff[:, [0]].reshape(-1) +
    #         max_out * mid_diff[:, [1]].reshape(-1),
    #         dtype=np.float32
    #     )
    #     out = out.reshape([N, out_h, out_w, C])  # nhwc
    #     return np.array(out, dtype=np.float32)

    def fc(self, inputs):
        assert type(inputs) == type(np.array([0.]))
        inputs = np.transpose(inputs, [0, 3, 1, 2])  # nchw
        N, C, H, W = inputs.shape
        out_h = (H + 2 * self.pad - self.pool_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_size) // self.stride + 1
        # inputs as nchw's ndarray
        col, diff_col = self.im2col(
            inputs, self.pool_size, self.pool_size, self.stride, self.pad)
        zeros = np.zeros_like(diff_col)
        ones = np.ones_like(diff_col)
        # pix = 1 if pix > mean(X) else 0
        diff_col = np.where(diff_col > np.mean(diff_col), ones, zeros)
        # n*oh*ow,c*fh*fw -> n*oh*ow*c,fh*fw
        col = col.reshape(-1, self.pool_size * self.pool_size)
        # n*oh*ow,c*fh*fw -> n*oh*ow*c,fh*fw
        diff_col = diff_col.reshape(-1, self.pool_size * self.pool_size)
        # n*oh*ow*c,1 ?????
        mid_diff = diff_col[:, [self.pool_size * self.pool_size // 2]]
        # (n*oh*ow*c,)
        mid_diff = mid_diff.reshape(-1).astype(np.uint8)  # 向下取整
        # (n*oh*ow*c,2)
        mid_diff = np.eye(2)[mid_diff]
        mean_out = np.mean(col, axis=1).reshape(-1)
        max_out = np.max(col, axis=1).reshape(-1)
        out = np.array(
            mean_out * mid_diff[:, [0]].reshape(-1) +
            max_out * mid_diff[:, [1]].reshape(-1),
            dtype=np.float32
        )
        out = out.reshape([N, C, out_h, out_w])  # nchw
        return np.array(out, dtype=np.float32)

    # def compute_output_shape(self, input_shape):
    #     N, H, W, C, = input_shape
    #     out_h = (H + 2 * self.pad - self.pool_size) // self.stride + 1
    #     out_w = (W + 2 * self.pad - self.pool_size) // self.stride + 1
    #     return [N, out_h, out_w, C]

    def compute_output_shape(self, input_shape):
        N, C, H, W, = input_shape
        out_h = (H + 2 * self.pad - self.pool_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_size) // self.stride + 1
        return [N, C, out_h, out_w]

    # def call(self, inputs, **kwargs):
    #     # out = Ax
    #     out = tf.numpy_function(
    #         func=self.fc,
    #         inp=[inputs],
    #         Tout=[tf.float32]

    #     )
    #     # reshape
    #     # h = ( h + 2*pad - pool_size) // self.stride + 1
    #     shape = inputs.get_shape().as_list() # nhwc
    #     out_shape = (shape[1] + 2 * self.pad -
    #                  self.pool_size) // self.stride + 1
    #     out = tf.reshape(out, shape=[-1, out_shape, out_shape, shape[3]])
    #     return out

    def forward(self, inputs, **kwargs):
        # ????
        out = self.fc(inputs.detach().numpy())  # (256, 28, 17, 15)
        shape = list(inputs.shape)  # nchw # [256, 32, 28, 28]
        out_shape = (shape[-2] + 2 * self.pad -
                     self.pool_size) // self.stride + 1  # 15
        out = np.reshape(out, [-1, shape[1], out_shape, out_shape])
        print(out.shape)
        exit()
        return out

# 数据生成器 todo
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train[:, :, :, np.newaxis].astype(np.float32) / 255.0
# x_test = x_test[:, :, :, np.newaxis].astype(np.float32) / 255.0
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #    transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True)

# 构建模型 todo
# conv311 - gp - bn - relu - conv311 - gp - bn - relu - gap - dense(10) - softmax
# inputs = tf.keras.Input(shape=(28, 28, 1))
# x = tf.keras.layers.Conv2D(  # 28*28
#     filters=32,
#     kernel_size=3,
#     padding='same'
# )(inputs)
# x = GradPooling_v2(  # 14*14
#     pool_size=2,
#     stride=2,
#     pad=1
# )(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
# x = tf.keras.layers.Conv2D(  # 14*14
#     filters=32,
#     kernel_size=3,
#     padding='same'
# )(x)
# x = GradPooling_v2(  # 7*7
#     pool_size=2,
#     stride=2,
#     pad=1
# )(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dense(units=10)(x)
# outputs = tf.keras.layers.Activation(
#     activation=tf.keras.activations.softmax)(x)

# model = tf.keras.models.Model(inputs=inputs, outputs=outputs)


def convs(inc, outc, ks, stri, padd, pool=None, norm=None, act=None):
    m = [nn.Conv2d(inc, outc, ks, stri, padd)]
    m += [eval(pool)] if pool is not None else []
    m += [eval(norm)] if norm is not None else []
    m += [eval(act)] if act is not None else []
    return nn.Sequential(*m)


## conv311 - gp - bn - relu - conv311 - gp - bn - relu - gap - dense(10) - softmax
model = nn.Sequential(
    # inputs: n,1,28,28
    convs(1, 32, 3, 1, 1,
          pool="GradPooling_v2(pool_size=2, stride=2, pad=1)",  # 14*14
          norm="nn.BatchNorm2d(num_features=32)",
          act="nn.ReLU()"),
    convs(32, 32, 3, 1, 1,
          pool="GradPooling_v2(pool_size=2, stride=2, pad=1)",  # 7*7
          norm="nn.BatchNorm2d(num_features=32)",
          act="nn.ReLU()",),
    nn.AdaptiveAvgPool2d(1),
    models.DenseNet(num_classes=10),  # 模型参数！
    nn.Softmax2d(),
)

# 训练策略 todo
# adam - crossentropy - crosstropy&accuracy
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss=tf.keras.losses.categorical_crossentropy,
#     metrics=[tf.keras.metrics.categorical_crossentropy,
#              tf.keras.metrics.categorical_accuracy]
# )
# # model.summary() # 打印参数
# # 训练网络
# model.fit(
#     x=x_train,
#     y=y_train,
#     batch_size=batch_size,
#     validation_split=0.1,
#     shuffle=True,
#     epochs=600
# )


def train(net: nn.Module, device, train_loader, eval_loader, optimizer: optim.Adam, lr, loss: nn.CrossEntropyLoss, metrics=[...]):
    net = net.to(device).train()
    optimizer = optimizer(net.parameters(), lr)
    loss = loss()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        l = loss(y, y_hat)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    net.eval()
    metrics_sum = np.array([None] * len(metrics))
    for x, y in eval_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_hat = net(x)
        metric_new = np.array([m(y, y_hat) for m in metrics])
        metrics_sum += metric_new

    metrics_mean = tuple(metrics_sum / eval_loader.__len__())
    print("metrics:", metrics_mean)


def crossentropyloss(y_hat, y):
    return F.cross_entropy(y_hat, y).item()


def accuracy(y_hat, y, num):
    return (y.eq(y_hat.data.max(1)[1]).sum() / num).item()


train(model, device, train_loader, test_loader,
      optim.Adam, 1e-3, nn.CrossEntropyLoss, metrics=[crossentropyloss, accuracy])
