import tensorflow as tf
import numpy as np

class percLoss(tf.keras.losses.Loss):

	# def __init__(self):
	# 	super.__init__()

	def call(self, y_true, y_pred):

		# print("Shape:" , y_pred.get_shape())
		# print("y_pred is: ", y_pred)

		# print(len(y_true))
		
		# tf.print(len(y_true[0]))

		# percTrue = tf.constant([[0.8]])
		# percTrue = y_true
		tf.print("y_pred shape:", y_pred.shape)
		tf.print("y_true shape:", y_true.shape)
		# tf.print("y_true[0] shape:", y_true[0].shape)

		tf.print("y_true:", y_true)

		# y_predA = y_pred[:,:,:,0]

		# tf.print("y_predA:", y_predA)


		# y_predB = y_pred[:,:,:,1]
		# tf.print("y_predA shape:", tf.math.greater(y_predA, tf.constant([0.9])))
		# y_predB = tf.math.greater(y_predA, tf.constant([0.9]))
		# tf.print("y_predB:", y_predB)
		# tf.print("y_predB shape:", y_predB.shape)
		# y_predB = y_pred[:,:,:,1]

		# print(y_predA.get_shape())
		# print(y_predB.get_shape())

		# print(tf.reduce_sum(y_predA))
		# print(percTrue)

		# percPred = tf.cast(tf.reduce_sum(y_predA), dtype=tf.float32) / tf.cast(tf.size(y_predA), dtype=tf.float32)

		# percPred = tf.constant([[0.8],[0.8],[0.8],[0.8]])

		y_pred = tf.cast(tf.reduce_sum(tf.cast(tf.math.greater(y_pred[:,:,:,0], tf.constant([0.9])), dtype=tf.int32), axis = [1,2]), dtype=tf.float32) / tf.cast(tf.size(y_pred[:,:,:,0]), dtype=tf.float32)

		# percPred2 = tf.cast(tf.reduce_sum(y_predB), dtype=tf.float32) / tf.cast(tf.size(y_predB), dtype=tf.float32)
		# tf.print("PercPred:", percPred)
		# tf.print("PercPred2:", percPred2)
		# tf.print("PercPred Sum:", percPred + percPred2)

		mae = tf.keras.losses.MeanAbsoluteError()

		tf.print("Batch MAE:", mae(y_true, y_pred))

		return mae(y_true, y_pred)