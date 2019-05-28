#coding=utf-8
#Learn Blog:https://blog.csdn.net/qq_26499769/article/details/83831640
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# 数据输入
mnist = input_data.read_data_sets('MNIST', one_hot=True)

# 创建保存模型文件的文件夹
if not os.path.exists("logdir"):
    os.makedirs("logdir")
LOGDIR = "logdir"

# 定义超参数
real_img_size = mnist.train.images[0].shape[0]
noise_size = 100
noise = 'normal0-1'
learning_rate = 0.001
batch_size = 100
epochs = 120

# 定义leakyRel激活函数
def leakyRelu(x, alpha=0.01):
    return tf.maximum(x, alpha*x)

# 输入数据占位
def get_inputs(real_img_size, noise_size):
    real_img = tf.placeholder(tf.float32, shape=[None, real_img_size], name = "real_img")
    real_img_digit = tf.placeholder(tf.float32, shape=[None, 10])
    noise_img = tf.placeholder(tf.float32, shape=[None, noise_size], name = "noise_img")
    return real_img, noise_img, real_img_digit

# 全连接层
def fully_connected(name, value, output_shape):
    with tf.variable_scope(name, reuse=None) as scope:
        shape = value.get_shape().as_list()
        w = tf.get_variable('w', [shape[1], output_shape], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable('b', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        return tf.matmul(value, w) + b

# 产生加性噪声
def get_noise(noise, batch_size):
    if noise == 'uniform':
        batch_size = np.random.uniform(-1, 1, size=(batch_size, noise_size)) #从均匀分布[low, high)中采样
    elif noise == 'normal':
        batch_size = np.random.normal(-1, 1, size=(batch_size, noise_size)) #高斯分布，参数分别为，均值，标准差，输出的shape
    elif noise == 'normal0-1':
        batch_noise = np.random.normal(0, 1, size=(batch_size, noise_size))
    elif noise == 'uniform0-1':
        batch_size = np.random.uniform(0, 1, size=(batch_size, noise_size))
    return batch_noise

# 构造生成器
def get_generator(digit, noise_img, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        concatenated_img_digit = tf.concat([digit, noise_img], 1)
        output = fully_connected('gf1', concatenated_img_digit, 128)
        output = leakyRelu(output)
        output = tf.layers.dropout(output, rate=0.5)

        output = fully_connected('gf2', output, 128)
        output = leakyRelu(output)
        output = tf.layers.dropout(output, rate=0.5)

        logits = fully_connected('gf3', output, 784)
        outputs = tf.tanh(logits)
        return logits, outputs

# 构造鉴别器
def get_discriminator(digit, img, reuse = False):
    with tf.variable_scope("discriminator", reuse=reuse):
        concatenated_img_digit = tf.concat([digit, img], 1)
        output = fully_connected('df1', concatenated_img_digit, 128)
        output = leakyRelu(output)
        output = tf.layers.dropout(output, rate=0.5)

        output = fully_connected('df2', output, 128)
        output = leakyRelu(output)
        output = tf.layers.dropout(output, rate=0.5)

        logits = fully_connected('df3', output, 1)
        output = tf.sigmoid(logits)
        return logits, output

# 保存生成器产生的数字
def save_genImages(gen, epoch):
    r, c = 10, 10
    fig, axs = plt.subplots(r, c)
    cnt = 0
#    print(gen.shape)
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen[cnt][:,:], cmap='Greys_r')
            axs[i, j].axis('off')
            cnt += 1
    if not os.path.exists('gen_mnist'):
        os.makedirs('gen_mnist')
    fig.savefig('gen_mnist/%d.jpg' % epoch)
    plt.close()

# 保存loss曲线
def  plot_loss(loss):
    fig, ax = plt.subplots(figsize=(20, 7))
    losses = np.array(loss)
    plt.plot(losses.T[0], label='Discriminator Loss')
    plt.plot(losses.T[1], label='Discriminator_real_loss')
    plt.plot(losses.T[2], label='Discriminator_fake_loss')
    plt.plot(losses.T[3], label='Generator Loss')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('loss1.jpg')
    plt.show()

# 保存损失函数的值
def Save_lossValue(e, epochs, train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g):
    with open('loss1.txt', 'a') as f:
        f.write("Epoch {}/{}".format(e+1, epochs), "Discriminator loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})".
                format(train_loss_d, train_loss_d_real, train_loss_d_fake), "Generator loss: {:.4f}".format(train_loss_d))

#清除每次运行时,tensorflow不断增加的节点并重置整个default graph
tf.reset_default_graph()

real_img, noise_img, real_img_digit = get_inputs(real_img_size, noise_size)

# 生成器
g_logits, g_outputs = get_generator(real_img_digit, noise_img)
sample_images = tf.reshape(g_outputs, [-1, 28, 28, 1])
tf.summary.image("sample_images", sample_images, 10) #10代表要生成图像的最大批处理元素数
# 判别器
d_logits_real, d_outputs_real = get_discriminator(real_img_digit, real_img)
d_logits_fake, d_outputs_fake = get_discriminator(real_img_digit, g_outputs, reuse=True)
# 判别器损失
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
d_loss = tf.add(d_loss_fake, d_loss_real)

# 生成器损失
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

#tesnorboard序列化损失
tf.summary.scalar("d_loss_real", d_loss_real) #用来显示标量信息
tf.summary.scalar("d_loss_fake", d_loss_fake)
tf.summary.scalar("d_loss", d_loss)
tf.summary.scalar("g_loss", g_loss)

# 分别训练生成器和判别器
# optimizer
train_vars = tf.trainable_variables()
# generator tensor
g_vars = [var for var in train_vars if var.name.startswith("generator")]
#discriminator tensor
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
# optimizer
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

summary = tf.summary.merge_all() #将所有的summary全部保存到磁盘

saver = tf.train.Saver()
def train():
    #保存loss值
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
        for e in range(epochs):
            for i in range(mnist.train.num_examples//(batch_size * 10)):
                for j in range(10):
                    batch = mnist.train.next_batch(batch_size)
                    digits = batch[1]
                    images = batch[0].reshape((batch_size, 784))
                    images = 2 * images - 1 #生成器激活函数tanh(-1,1)，将原始图像(0-1)也变为(-1,1)
                    noises = get_noise(noise, batch_size)
                    sess.run([d_train_opt, g_train_opt], feed_dict={real_img:images, noise_img:noises, real_img_digit:digits})

            #训练损失
            summary_str, train_loss_d_real, train_loss_d_fake, train_loss_g = sess.run([summary, d_loss_real, d_loss_fake, g_loss],
                                                                                       feed_dict={real_img : images, noise_img : noises, real_img_digit : digits})
            train_loss_d = train_loss_d_fake + train_loss_d_real
            losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))
            summary_writer.add_summary(summary_str, e)
            summary_writer.flush()
            print("Epoch {}/{}".format(e+1, epochs), "Discriminator loss : {:.4f}(Real: {:.4f} + Fake: {:.4f})".format(train_loss_d,
                                                                                                                       train_loss_d_real, train_loss_d_fake),
                  "Generator loss: {:.4f}".format(train_loss_g))
            #保存模型
            saver.save(sess, 'checkpoints/cgan.ckpt')
            #查看每轮结果
            gen_sample = get_noise(noise, batch_size)
            lable = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] * batch_size #给定标签条件生成指定的数字
            labels = np.array(lable)
            labels = labels.reshape(-1, 10)
            _, gen = sess.run(get_generator(real_img_digit, noise_img, reuse=True), feed_dict={noise_img:gen_sample, real_img_digit:labels})
            if e % 1 == 0:
                gen = gen.reshape(-1, 28, 28)
                gen = (gen + 1) / 2 #拉回到原来取值范围
                save_genImages(gen, e)
        plot_loss(losses)

def test():
    saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        saver.restore(sess, 'checkpoints/cgan.ckpt')
        sample_noise = get_noise(noise, batch_size)
        label = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]*100
        labels = np.array(label)
        labels = labels.reshape(-1, 10)
        _, gen_samples = sess.run(get_generator(real_img_digit, noise_img, reuse=True), feed_dict={noise_img:sample_noise, real_img_digit:labels})
        for i in range(len(gen_samples)):
            plt.imshow(gen_samples[i].reshape(28, 28), cmap='Greys_r')
            plt.show()

if __name__ == '__main__':
    train()