import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from pathlib import Path
import os
import csv


class OxfordPets(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y


# class OxfordPets(tf.keras.utils.Sequence):
#     """Helper to iterate over the data (as Numpy arrays)."""

#     def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.input_img_paths = input_img_paths
#         self.target_img_paths = target_img_paths
#         self.rootDir = Path(__file__).parent.parent

#         # Create a dict from the csv with the labels?
#         self.percgt = dict()

#         with open(os.path.join(self.rootDir.parent,'percs.csv'), 'r') as csvfile:
#         	csvreader = csv.reader(csvfile)

#         	for lines in csvreader:
#         		# Do things
#         		k = lines[0]
#         		v1 = float(lines[1])
#         		v2 = float(lines[2])
#         		self.percgt[k] = v2


#     def __len__(self):
#         return len(self.target_img_paths) // self.batch_size

#     def __getitem__(self, idx):
#         """Returns tuple (input, target) correspond to batch #idx."""
#         i = idx * self.batch_size
#         batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
#         batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
#         x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
#         for j, path in enumerate(batch_input_img_paths):
#             img = load_img(path, target_size=self.img_size)
#             x[j] = img
#         y = np.zeros((self.batch_size,) + (1,), dtype="float32")
#         for j, path in enumerate(batch_target_img_paths):
#             # img = load_img(path, target_size=self.img_size, color_mode="grayscale")
#             # y[j] = np.expand_dims(img, 2)
#             pth = path.split('Thesis_Work')[1]
#             # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
#             y[j] = self.percgt[pth]
#         return x, y
