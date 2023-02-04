import tensorflow as tf

def create_mask(pred_mask):
	pred_mask = tf.argmax(pred_mask, axis=-1)
	pred_mask = pred_mask[..., tf.newaxis]
	return pred_mask[0]

def show_predictions(dataset=None, num=1):
 if dataset:
	for image, mask in dataset.take(num):
	pred_mask = unet_model.predict(image)
	display([image[0], mask[0], create_mask(pred_mask)])
 else:
	display([sample_image, sample_mask,
		create_mask(model.predict(sample_image[tf.newaxis, ...]))])

count = 0
for i in test_batches:
	count +=1
print("number of batches:", count)