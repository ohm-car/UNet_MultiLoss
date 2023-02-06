import tensorflow as tf

class percLoss(tf.keras.losses.Loss):

	def __init__(self):
		super.__init__()

	def call(self, y_true, y_pred):

		percTrue = ?
		percPred = ?

		mse = tf.keras.losses.MeanSquaredError()

		return mse(percTrue, percPred)