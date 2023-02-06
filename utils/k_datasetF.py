import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io


class DatasetUSound(object):
	"""docstring for ClassName"""
	def __init__(self):
		super(DatasetUSound, self).__init__()
		
		# self.load_dataset()

		# print(self.dataset)
		
		# self.arg = arg
		

	def load_dataset(self):

		imgpath = os.path.abspath('../data/Img_All_Squared/')
		maskpath = os.path.abspath('../data/Masks_All_Squared/')
		# print(path)

		images = []
		masks = []

		for i in range(2, 92):

			imgs = []
			for j in range(3):
				img = skimage.io.imread(imgpath + '/' + str(i + j - 2) + '.png')
				imgs.append(img)
			images.append(imgs)

			mask = skimage.io.imread(maskpath + '/' + str(i) + '.png')
			mask = np.sum(mask, axis = 2) == 765
			mask = np.expand_dims(mask, axis=2)
			masks.append(mask)

		images = np.array(images)
		masks = np.array(masks)
		print('images.shape:', images.shape)
		print('masks.shape:', masks.shape)

		return images, masks



	def resize(self, input_image, input_mask):
		input_image = tf.image.resize(input_image, (128, 128), method="nearest")
		input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")

		return input_image, input_mask

	def augment(self, input_image, input_mask):
		if tf.random.uniform(()) > 0.5:
			# Random flipping of the image and mask
			input_image = tf.image.flip_left_right(input_image)
			input_mask = tf.image.flip_left_right(input_mask)

		return input_image, input_mask

	def normalize(self, input_image, input_mask):
		input_image = tf.cast(input_image, tf.float32) / 255.0
		input_mask -= 1
		return input_image, input_mask

	def load_image_train(self, datapoint):
		input_image = datapoint["image"]
		input_mask = datapoint["segmentation_mask"]
		input_image, input_mask = self.resize(input_image, input_mask)
		input_image, input_mask = self.augment(input_image, input_mask)
		input_image, input_mask = self.normalize(input_image, input_mask)

		return input_image, input_mask

	def load_image_test(self, datapoint):
		input_image = datapoint["image"]
		input_mask = datapoint["segmentation_mask"]
		input_image, input_mask = self.resize(input_image, input_mask)
		input_image, input_mask = self.normalize(input_image, input_mask)

		return input_image, input_mask

	def get_train_dataset(self):
		train_dataset = self.dataset["train"].map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
		print(type(train_dataset))
		return train_dataset

	def get_test_dataset(self):
		test_dataset = self.dataset["test"].map(self.load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
		print(type(test_dataset))
		return test_dataset

	def get_info(self):
		return self.info

# train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
# test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

# BATCH_SIZE = 64
# BUFFER_SIZE = 1000
# train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
# train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# validation_batches = test_dataset.take(3000).batch(BATCH_SIZE)
# test_batches = test_dataset.skip(3000).take(669).batch(BATCH_SIZE)