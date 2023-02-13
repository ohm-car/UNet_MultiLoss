import tensorflow as tf
import numpy as np

class percLoss2(tf.keras.losses.Loss):

	# def __init__(self):
	# 	super.__init__()

	def call(self, y_true, y_pred):

		# print("Shape:" , y_pred.get_shape())
		# print("y_pred is: ", y_pred)

		# print(len(y_true))
		
		# tf.print(len(y_true[0]))

		percTrue = tf.constant([[0.8]])
		# percTrue = y_true
		# tf.print("y_pred shape:", y_pred.shape)
		# tf.print("y_true shape:", y_true.shape)
		# tf.print("y_true[0] shape:", y_true[0].shape)

		# tf.print("y_true:", y_true[0])

		y_predA = y_pred[:,:,:,0]
		# tf.print("y_predA shape:", y_predA.shape)
		# y_predB = y_pred[:,:,:,1]

		# print(y_predA.get_shape())
		# print(y_predB.get_shape())

		# print(tf.reduce_sum(y_predA))
		# print(percTrue)

		percPred = tf.cast(tf.reduce_sum(y_predA), dtype=tf.float32) / tf.cast(tf.size(y_predA), dtype=tf.float32)
		# tf.print("PercPred:", percPred)

		mae = tf.keras.losses.MeanAbsoluteError()

		# tf.print("Batch MAE:", mae(percTrue, percPred))

		return mae(percTrue, percPred)