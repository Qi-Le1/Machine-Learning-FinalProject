import torch
import numpy as np

# a = torch.ones(5)
# b = a.numpy().copy()
# b=b+1
# print(a)
# print(b)
#
# print(a.type())
# print(b.type())
# a = np.ones(5)
# b = torch.from_numpy(a)
# print("b之前",b)
# np.add(a, 1, out=a)
# #a = a+1
# #a = a+1
# print(a)
# print("b之后",b)
x = torch.randn(1)
y = torch.randn(1)
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # a CUDA device object
#     y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
#     x = x.to(device)                       # or just use strings ``.to("cuda")``
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x)
    print(y)
    x + y
# x = torch.Tensor(5, 3)  # 构造一个未初始化的5*3的矩阵
# x = torch.rand(5, 3)  # 构造一个随机初始化的矩阵
# print(x) # 此处在notebook中输出x的值来查看具体的x内容
# print(x.type())
#
# z = torch.Tensor(1)
# print("z",z.item())
# print(z.type())
#
# print("--------")
# print(x.size())
#
# #NOTE: torch.Size 事实上是一个tuple, 所以其支持相关的操作*
# y = torch.rand(5, 3)
#
# #此处 将两个同形矩阵相加有两种语法结构
# print(x + y) # 语法一
# print(torch.add(x, y)) # 语法二
# print(y.add_(x))
# # 另外输出tensor也有两种写法
# result = torch.Tensor(5, 3) # 语法一
# torch.add(x, y, out=result) # 语法二
# y.add_(x) # 将y与x相加

# 特别注明：任何可以改变tensor内容的操作都会在方法名后加一个下划线'_'
# 例如：x.copy_(y), x.t_(), 这俩都会改变x的值。

# 另外python中的切片操作也是资次的。
# x[:,1] #这一操作会输出x矩阵的第二列的所有值


# part = [0,1,2]
# for i in range(3,0,-1):
#     print(i)
#
# print(part[:2])
# print(part[2:])
# print(part[2:])

import numpy as np
import tensorflow as tf

# a=np.ones((4,3,2,1))
#
# b = a[:,np.newaxis]
#
# print(b)
# print(b.shape)
#
# w = tf.get_variable('weight', shape=[3, 2], initializer=tf.initializers.random_normal())
# print(w)
# for i in range(10):
#     a = i // 2 + 2
#     print("第一行", i // 2 + 2)
#     b = 2**a
#     print("第二行", b)
# noise = tf.get_variable('noise', shape=[1,1,4,4], initializer=tf.initializers.random_normal(), trainable=False)
# variable_names = [v.name for v in noise]
# print(variable_names)
# weight = tf.get_variable('weight', shape=[512], initializer=tf.initializers.zeros())
# c = tf.reshape(tf.cast(weight, 'float32'), [1, -1, 1, 1])
# a = noise * tf.reshape(tf.cast(weight, 'float32'), [1, -1, 1, 1])
# a
#
# def G_wgan_div(G, D, opt, training_set, minibatch_size):  # pylint: disable=unused-argument
#     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     labels = training_set.get_random_labels_tf(minibatch_size)
#     fake_images = G.get_output_for(latents, labels, is_training=True)
#     fake_scores = fp32(D.get_output_for(fake_images, labels, is_training=True))
#     loss = -fake_scores
#     return loss
#
# def D_wgan_div(G, D, opt, training_set, minibatch_size, reals, labels,  # pylint: disable=unused-argument
#     wgan__div_lambda=2.0,  # Weight for the gradient penalty term.
#     wgan_div_epsilon=0.001,  # Weight for the epsilon term, \epsilon_{drift}.
#     wgan_div_target=1.0):  # Target value for gradient magnitudes.
#     wgan_div_power = 6
#
#     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     fake_images = G.get_output_for(latents, labels, is_training=True)
#     real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
#     fake_scores = fp32(D.get_output_for(fake_images, labels, is_training=True))
#     real_scores_out = autosummary('Loss/scores/real', real_scores_out)
#     fake_scores = autosummary('Loss/scores/fake', fake_scores)
#     loss = fake_scores - real_scores_out
#
#     with tf.name_scope('GradientPenalty'):
#         mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images.dtype)
#         mixed_images_out = tflib.lerp(tf.cast(reals, fake_images.dtype), fake_images, mixing_factors)
#         mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
#         mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
#         mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
#         mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
#         mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1, 2, 3]))
#         mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
#         gradient_penalty = tf.pow(mixed_norms, wgan_div_power)
#     loss += gradient_penalty * (wgan__div_lambda / (wgan_div_target ** 2))
#
#     with tf.name_scope('EpsilonPenalty'):
#         epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
#     loss += epsilon_penalty * wgan_div_epsilon
#     return loss
#
#
#     with tf.name_scope('EpsilonPenalty'):
#         epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
#     loss += epsilon_penalty * wgan_div_epsilon
