import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

class UNet(object):
	"""docstring for ClassName"""
	def __init__(self):

		super(UNet, self).__init__()
		# self.unet_model = self.create_model()
		# print(self.unet_model.summary())
		# self.args = 0
		# self.arg = arg
		
	def double_conv_block(self, x, n_filters):

		# Conv2D then ReLU activation
		x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
		# Conv2D then ReLU activation
		x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

		return x

	def downsample_block(self, x, n_filters):
		f = self.double_conv_block(x, n_filters)
		p = layers.MaxPool2D(2)(f)
		p = layers.Dropout(0.3)(p)

		return f, p

	def upsample_block(self, x, conv_features, n_filters):
		# upsample
		x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
		# concatenate
		x = layers.concatenate([x, conv_features])
		# dropout
		x = layers.Dropout(0.3)(x)
		# Conv2D twice with ReLU activation
		x = self.double_conv_block(x, n_filters)

		return x

	def create_common_submodel(self):

		# inputs
		inputs = layers.Input(shape=(160,160,3))

		# encoder: contracting path - downsample
		# 1 - downsample
		f1, p1 = self.downsample_block(inputs, 64)
		# 2 - downsample
		f2, p2 = self.downsample_block(p1, 128)
		# 3 - downsample
		f3, p3 = self.downsample_block(p2, 256)
		# 4 - downsample
		f4, p4 = self.downsample_block(p3, 512)

		# 5 - bottleneck
		bottleneck = self.double_conv_block(p4, 1024)

		# decoder: expanding path - upsample
		# 6 - upsample
		u6 = self.upsample_block(bottleneck, f4, 512)
		# 7 - upsample
		u7 = self.upsample_block(u6, f3, 256)
		# 8 - upsample
		u8 = self.upsample_block(u7, f2, 128)
		# 9 - upsample
		u9 = self.upsample_block(u8, f1, 64)

		# outputs
		# outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)

		# unet model with Keras Functional API
		unet_sub_model = tf.keras.Model(inputs, u9, name="U-Net")

		# print(unet_sub_model.summary())

		return unet_sub_model
		# return u9

	def get_reconstructed_output_image(self, out1):

		image = layers.Conv2D(3, 1, padding = "same", activation = "sigmoid", name = "image_output")(out1)
		return image

	def get_percentages_output(self, out1):
		
		mask = layers.Conv2D(2, 1, padding = "same", activation = "softmax", name = "mask_output")(out1)
		return mask

	def build_model(self):

		inputs = layers.Input(shape=(160,160,3))
		unet_sub_model = self.create_common_submodel()
		image = self.get_reconstructed_output_image(unet_sub_model(inputs))
		mask = self.get_percentages_output(unet_sub_model(inputs))
		unet_model = tf.keras.Model(inputs = inputs, outputs = [image, mask],
			name = "UNetMultiLoss")
		# print(unet_model.summary())

		return unet_model


# print("Test model")
# model = create_model()
# print(model.summary())
# print("Done")