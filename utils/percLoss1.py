import tensorflow as tf

class percLoss(tf.keras.losses.Loss):

	# def __init__(self):
	# 	super.__init__()

	def call(self, y_true, y_pred):

		print("Shape:" , y_pred.get_shape())
		print("y_pred is: ", y_pred)

		percTrue = tf.constant([[0.5]])
		
		y_predA = y_pred[:,:,:,0]
		# y_predB = y_pred[:,:,:,1]

		print(y_predA.get_shape())
		# print(y_predB.get_shape())

		print(tf.reduce_sum(y_predA))
		print(percTrue)

		percPred = tf.cast(tf.reduce_sum(y_predA), dtype=tf.float32) / tf.cast(tf.size(y_predA), dtype=tf.float32)

		mae = tf.keras.losses.MeanAbsoluteError()

		return mae(percTrue, percPred)