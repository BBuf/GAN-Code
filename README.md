# GAN-Code

- GAN.pdf 原始的对抗生成网络论文。

- GAN.py 原始GAN的tensorflow实现，细节可以查看：https://blog.csdn.net/just_sort/article/details/79454054 ,运行80个epoch生成的结果为：

![](image/GAN.jpg)

- DCGAN.pdf 卷积对抗生成网络的论文原文。

- DCGAN.py 使用Keras实现DCGAN生成MNIST数据集，训练和测试方法如下：

  ```python
  训练： python dcgan.py --mode train --batch_size <batch_size>
  测试： python dcgan.py --mode generate --batch_size <batch_size> --nice
  ```

- DCGAN_face.py 使用Tensorflow实现DCGAN生成人脸，数据集的格式为：

![数据集](image/GAN_face_data.jpg) ，

细节可以查看：https://blog.csdn.net/just_sort/article/details/84581400

- SGAN.pdf 《Semi-Supervised Learning with Generative Adversarial Networks》论文
- SGAN.py SGAN的tensorflow实现，细节可以查看：https://blog.csdn.net/just_sort/article/details/90605197 ，每个batch真实样本占比0.1运行1000个epoch的生成结果为：
![SSGAN](image/ssgan-1000epoch.jpg)
- CGAN.pdf 条件GAN的论文原文。
- CGAN.py 使用Tensorflow实现CGAN进行特定类别的手写数字。


