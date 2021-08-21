import tensorflow as tf
import numpy as np
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
batch_size = 256

class GradPooling_v2(tf.keras.layers.Layer):

    def __init__(self, pool_size, stride, pad, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.pool_size = pool_size
        self.cs = self.pool_size // 2
        self.stride = stride
        self.arg_max = None
        self.pad = pad

    def build(self, input_shape):
        self.lamb = self.add_weight(name='lamb', shape=[1])
        return super().build(input_shape)

    def compute_diff(self, inputs):
        diff_x_left = inputs[:, :, 2:, 2:] - inputs[:, :, :-2, 2:]
        diff_x_right = inputs[:, :, :-2, 2:] - inputs[:, :, 2:, 2:]
        diff_y_top = inputs[:, :, 2:, 2:] - inputs[:, :, 2:, :-2]
        diff_y_bottom = inputs[:, :, 2:, :-2] - inputs[:, :, 2:, 2:]
        diffs = tf.abs(diff_x_right) + tf.abs(diff_y_bottom) + \
            tf.abs(diff_x_left) + tf.abs(diff_y_top)
        return diffs

    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_data.shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1
        img = np.pad(input_data, [(0, 0), (0, 0),
                                  (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
        diffs = self.compute_diff(img)
        diff_col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
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
        col = col.transpose([0, 4, 5, 1, 2, 3]).reshape(N * out_h * out_w, -1)
        diff_col = diff_col.transpose(
            [0, 4, 5, 1, 2, 3]).reshape(N * out_h * out_w, -1)
        return col, diff_col

    def fc(self, inputs):
        inputs = np.transpose(inputs, [0, 3, 1, 2])
        N, C, H, W = inputs.shape
        out_h = (H + 2 * self.pad - self.pool_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_size) // self.stride + 1
        col, diff_col = self.im2col(
            inputs, self.pool_size, self.pool_size, self.stride, self.pad)
        zeros = np.zeros_like(diff_col)
        ones = np.ones_like(diff_col)
        diff_col = np.where(diff_col > np.mean(diff_col), ones, zeros)
        col = col.reshape(-1, self.pool_size * self.pool_size)
        diff_col = diff_col.reshape(-1, self.pool_size * self.pool_size)
        mid_diff = diff_col[:, [self.pool_size * self.pool_size // 2]]
        mid_diff = mid_diff.reshape(-1).astype(np.uint8)
        mid_diff = np.eye(2)[mid_diff]
        mean_out = np.mean(col, axis=1).reshape(-1)
        max_out = np.max(col, axis=1).reshape(-1)
        out = np.array(
            mean_out * mid_diff[:, [0]].reshape(-1) +
            max_out * mid_diff[:, [1]].reshape(-1),
            dtype=np.float32
        )
        out = out.reshape([N, out_h, out_w, C])
        return np.array(out, dtype=np.float32)

    def compute_output_shape(self, input_shape):
        N, H, W, C, = input_shape
        out_h = (H + 2 * self.pad - self.pool_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_size) // self.stride + 1
        return [N, out_h, out_w, C]

    def call(self, inputs, **kwargs):
        # 包装为tf操作 - 
        out = tf.numpy_function(
            func=self.fc,
            inp=[inputs],
            Tout=[tf.float32]
        )
        shape = inputs.get_shape().as_list()
        out_shape = (shape[1] + 2 * self.pad -
                     self.pool_size) // self.stride + 1  # 15
        out = tf.reshape(out, shape=[-1, out_shape, out_shape, shape[3]])
        return out


class GradPooling(tf.keras.layers.Layer):

    def __init__(self, pool_size, stride, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.pool_size = pool_size
        self.cs = self.pool_size // 2
        self.stride = stride

    def build(self, input_shape):
        self.lamb = tf.Variable(tf.random.uniform(
            shape=[1], minval=0, maxval=1), dtype=tf.float32, trainable=True)
        return super().build(input_shape)

    def compute_diff(self, inputs):
        diff_x_left = inputs[:, :, 1:, 1:] - inputs[:, :, :-1, 1:]
        diff_x_right = inputs[:, :, :-1, 1:] - inputs[:, :, 1:, 1:]
        diff_y_top = inputs[:, :, 1:, 1:] - inputs[:, :, 1:, :-1]
        diff_y_bottom = inputs[:, :, 1:, :-1] - inputs[:, :, 1:, 1:]
        diffs = tf.abs(diff_x_right) + tf.abs(diff_y_bottom) + \
            tf.abs(diff_x_left) + tf.abs(diff_y_top)
        return diffs

    def call(self, inputs, **kwargs):
        cs = self.cs
        stride = self.stride
        pool_size = self.pool_size
        shape = inputs.get_shape().as_list()

        def inner_loop(i, j, inner_n, image, diff, col):
            window = tf.slice(image, begin=[(i - cs) * stride, (j - cs) * stride],
                              size=[pool_size, pool_size])
            mean = tf.reduce_mean(window)
            max = tf.reduce_max(window)
            v = [[mean, max]]
            result = tf.squeeze(tf.matmul(v, tf.reshape(
                tf.one_hot(diff[i][j], 2), shape=[2, 1])))
            col = col.write(j - self.cs, result)
            j += 1
            return i, j, inner_n, image, diff, col

        def outer_loop(i, outer_n, inner_n, image, diff, output_image):
            col = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            j = self.cs
            i, j, inner_n, image, diff, col = tf.while_loop(
                cond=lambda i, j, inner_n, image, diff, col: (
                    j - cs) * stride < inner_n,
                body=inner_loop,
                loop_vars=[i, j, inner_n, image, diff, col],
                parallel_iterations=batch_size,
            )
            col = col.stack()
            output_image = output_image.write(i - self.cs, col)
            i += 1
            return i, outer_n, inner_n, image, diff, output_image

        def channel_loop(inp):
            image = inp[0]
            diff = inp[1]
            inner_n = shape[1]
            outer_n = shape[2]
            i = cs

            tf.dynamic_partition()

            output_image = tf.TensorArray(
                dtype=tf.float32, size=0, dynamic_size=True)
            i, outer_n, inner_n, image, diff, output_image = tf.while_loop(
                cond=lambda i, outer_n, inner_n, image, diff, output_image: (
                    i - cs) * stride < outer_n,
                body=outer_loop,
                loop_vars=[i, outer_n, inner_n, image, diff, output_image],
                parallel_iterations=batch_size,
            )
            output_image = output_image.stack()
            return output_image

        def batch_loop(inp):
            image = inp[0]
            diff = inp[1]
            image_with_all_channel = tf.map_fn(
                fn=channel_loop,
                elems=[image, diff],
                dtype=tf.float32
            )
            return image_with_all_channel
            pass

        padding_image = tf.keras.layers.ZeroPadding2D(
            padding=((1, 1), (1, 1)))(inputs)
        # NCWH
        padding_images = tf.transpose(padding_image, perm=[0, 3, 1, 2])
        diffs = self.compute_diff(padding_images)
        ones = tf.ones_like(diffs, dtype=tf.uint8)
        zeros = tf.zeros_like(diffs, dtype=tf.uint8)
        diffs = tf.where(diffs > self.lamb, ones, zeros)
        pool_result = tf.map_fn(
            fn=batch_loop,
            elems=[padding_images, diffs],
            dtype=tf.float32

        )
        result = tf.transpose(pool_result, perm=[0, 2, 3, 1])
        result = tf.reshape(
            result, shape=[-1, shape[1] // stride, shape[2] // stride, shape[3]])
        return result


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:, :, :, np.newaxis].astype(np.float32) / 255.0
x_test = x_test[:, :, :, np.newaxis].astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(  # 28*28
    filters=32,
    kernel_size=3,
    padding='same'
)(inputs)
x = GradPooling_v2(  # 14*14
    pool_size=2,
    stride=2,
    pad=1
)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
x = tf.keras.layers.Conv2D(  # 14*14
    filters=32,
    kernel_size=3,
    padding='same'
)(x)
x = GradPooling_v2(  # 7*7
    pool_size=2,
    stride=2,
    pad=1
)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Activation(
    activation=tf.keras.activations.softmax)(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_crossentropy,
             tf.keras.metrics.categorical_accuracy]
)
model.summary()
model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    validation_split=0.1,
    shuffle=True,
    epochs=600
)
scores = model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
print(scores)
