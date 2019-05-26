# GAN-Code

- GAN.pdf 原始的对抗生成网络论文。

- GAN.py 原始GAN的实现，细节可以查看：https://blog.csdn.net/just_sort/article/details/79454054

- DCGAN.pdf 卷积对抗生成网络的论文原文。

- DCGAN.py 使用Keras实现DCGAN生成MNIST数据集，训练和测试方法：

  ```python
  训练： python dcgan.py --mode train --batch_size <batch_size>
  测试： python dcgan.py --mode generate --batch_size <batch_size> --nice
  ```

- 