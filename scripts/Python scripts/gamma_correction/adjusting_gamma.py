from __future__ import print_function
import numpy as np
import argparse
import cv2
import glob
import shutil
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--value", type=float, required= True,
	help="# of training samples to generate")
args = vars(ap.parse_args())

def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

CATEGORIES = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
PATH = 'E:/IRELAND/MASTERS/PW27/Datasets/Use_dataset/Kaggle'
training_dir = PATH + '/validation'
modified_dir = 'E:/IRELAND/MASTERS/PW27/Datasets/Gamma/validation'
from tqdm import tqdm
for category in CATEGORIES:
      path1 = os.path.join(training_dir, category)
      path2 = os.path.join(modified_dir, category)
      class_num = CATEGORIES.index(category)
      for img in tqdm(os.listdir(path1)):
      	img1 = cv2.imread(os.path.join(path1, img))
      	gamma_image = adjust_gamma(img1, gamma=args["value"])
      	cv2.imwrite(os.path.join(path2, img), gamma_image)