import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
# from tensorflow import keras
import random
# from utils.k_dataset import DatasetTrial
# from unet.unet_model_k import UNet
from unet.unet_model_keras import UNet
from utils.k_dataset import OxfordPets
from utils.percLoss1 import percLoss
# import skimage
from pathlib import Path


rootDir = Path(__file__).parent
print(rootDir)

imageDir = os.path.join(rootDir.parent, 'data/images')
masksDir = os.path.join(rootDir.parent, 'data/annotations/trimaps')
# checkpoint_path = os.path.join(rootDir, 'checkpoints/model_{epoch:03d}')
checkpoint_path = os.path.join(rootDir, 'checkpointsUNet/model_{epoch:03d}')

# imageDir = '/nfs/ada/oates/users/omkark1/ArteryProj/data/Img_All_Squared/'
# masksDir = '/nfs/ada/oates/users/omkark1/ArteryProj/data/Masks_All_Squared/'
# checkpoint_path = "/nfs/ada/oates/users/omkark1/ArteryProj/UNetSeq/checkpointsR4/model_{epoch:03d}"

input_img_paths = sorted(
    [
        os.path.join(imageDir, fname)
        for fname in os.listdir(imageDir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(masksDir, fname)
        for fname in os.listdir(masksDir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))


# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
img_size = (160, 160)
num_classes = 3
batch_size = 8
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

model = UNet().build_model()
# model = UNet.get_model(img_size = img_size, num_classes = num_classes)

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
# model.compile(optimizer="rmsprop", loss=tf.keras.losses.SparseCategoricalCrossentropy())

losses = { "image_output" : tf.keras.losses.MeanSquaredError(),
            "mask_output" : percLoss()}

model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=losses,
                  metrics="accuracy")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path)
]

# Train the model, doing validation at the end of each epoch.
epochs = 40
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)