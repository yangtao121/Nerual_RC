import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class Network:
    def __init__(self, input_dims, output_dims, K, T_step, actor_lr):
        """

        @param actor_lr:
        @param input_dims:输入的维度
        @param output_dims:输出的维度
        @param K:神经网络的放大系数
        @param T_step:函数截时间戳
        """
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.K = K  # 动作系数值
        self.actor = self.actor_net()
        self.T_step = T_step  # 处理时间间隔
        self.counter = 0
        self.last_state = 0
        self.last_state_one = 0
        self.input_batch = np.zeros((self.T_step, self.input_dims))
        self.real_output_batch = np.zeros((self.T_step, self.output_dims))
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    def actor_net(self):
        """
        该网络为了后期框架升级采用actor_net名字
        """

        # 初始化权值
        last_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

        # U的输入
        u_inputs = layers.Input(shape=(self.input_dims,))
        u_out = layers.Dense(16, activation='relu')(u_inputs)
        u_out = layers.BatchNormalization()(u_out)
        u_out = layers.Dense(32, activation='relu')(u_out)
        u_out = layers.BatchNormalization()(u_out)

        # 上一个状态的输入
        last_input = layers.Input(shape=(self.output_dims,))
        last_out = layers.Dense(16, activation='relu')(last_input)
        last_out = layers.BatchNormalization()(last_out)
        last_out = layers.Dense(32, activation='relu')(last_input)
        last_out = layers.BatchNormalization()(last_out)

        concat = layers.Concatenate()([u_out, last_out])

        out = layers.Dense(512, activation='relu')(concat)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(512, activation='relu')(out)
        out = layers.BatchNormalization()(out)

        outputs = layers.Dense(self.output_dims, activation="tanh", kernel_initializer=last_init)(out)

        outputs = outputs * self.K
        model = tf.keras.Model([u_inputs, last_input], outputs)
        return model

    def get_batch(self, input_batch, real_output_batch):
        """
        存储相应的数据
        @param input_batch: 输入值
        @param real_output_batch: 产生的真实输出
        @return: 判断缓冲区是否满了
        """
        # 将数据存入数组中
        index = self.counter % self.T_step
        self.input_batch[index] = input_batch
        self.real_output_batch[index] = real_output_batch

        self.counter += 1  # 存储器次数加一

        # 判断缓冲器是否已经满了
        if index + 1 == self.T_step:
            flag = 1
        else:
            flag = 0
        return flag

    def get_last_state(self, last_state):
        """
        获取上一个状态值
        @param last_state: i-1时刻的真实值
        """
        self.last_state_one = last_state
        self.last_state = np.ones((self.T_step, self.input_dims)) * last_state

    def learn(self):
        """
        V1：在这个版本中，微分预测函数采用基线叠加法，适用于线性系统，对非线性系统支持较差
        @return:
        """
        tf_input_batch = tf.convert_to_tensor(self.input_batch, dtype=tf.float32)
        tf_real_output_batch = tf.convert_to_tensor(self.real_output_batch, dtype=tf.float32)
        tf_last_state = tf.convert_to_tensor(self.last_state, dtype=tf.float32)
        with tf.GradientTape() as tape:
            predict = self.actor([tf_input_batch, tf_last_state])
            # 线性叠加法
            y = tf_real_output_batch + tf_last_state - predict
            loss = tf.math.reduce_mean(tf.math.square(y))
        grad = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grad, self.actor.trainable_variables)
        )
        state = tf.squeeze(predict+tf_last_state)
        state = state.numpy()
        state = np.squeeze(state)
        return state

    def output_show(self, u_input):
        tf_last_state = tf.convert_to_tensor(self.last_state_one, dtype=tf.float32)
        out = tf.squeeze(self.actor([u_input, tf_last_state]))
        out = out.numpy()
        return out

    def save_model(self):
        self.actor.save_weights("actor.h5")
