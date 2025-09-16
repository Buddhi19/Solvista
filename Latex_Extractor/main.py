import os
import sys

sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
from model_load import for_test
from PIL import Image, ImageTk
import cv2
import imutils
import re

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"\\Latex_Extractor"

class Expressions:
	def __init__(self,img):
		self.img = img
		self.expressions = []

	def draw_contours(self):
		"""
		find different expressions or select the expression 
		written area
		"""
		kernel = np.ones((8,8),np.uint8)  #10,10

		dilation = cv2.dilate(self.img, kernel, iterations = 16) #16

		# dilation = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)

		contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		contours = [cnt for cnt in contours if (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3])>=0.50]

		#find the largest contour
		contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

		im2 = self.img.copy()
		print(contours)
		for cnt in contours:
			im2 = self.img.copy()
			x, y, w, h = cv2.boundingRect(cnt)
			
			rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 0)
			cropped = im2[y:y + h, x:x + w]
			
			scale_factor = 0.7
			cropped = cv2.resize(cropped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

			cv2.imshow("img",cropped)
			cv2.waitKey(1000)

			self.expressions.append(cropped)
		
		cv2.imshow("img",im2)
		cv2.waitKey(0)
		return
	
	def get_expressions(self):
		"""
		Get all expression containing images
		"""
		for image in self.expressions:
			cv2.imshow("img",image)
			cv2.waitKey(0)

		return self.expressions

class Image2Text:
	def pre_process(self,img):
		"""
		preprocess the img -> set a black background
		expressions in white
		"""
		img_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_, img_test = cv2.threshold(img_test, 200, 255, cv2.THRESH_BINARY) # 85 # 155
		img_test = cv2.bitwise_not(img_test)

		cv2.imshow("img",img_test)
		cv2.waitKey(0)

		# img_test = cv2.erode(img_test, np.ones((2,2),np.uint8), iterations = 3)

		# cv2.imshow("img",img_test)
		# cv2.waitKey(0)

		return img_test

	def model_eligible_format(self,img):
		"""
		make the img_test eligible for the model
		"""
		img_proceed = torch.from_numpy(np.array(img)).type(torch.FloatTensor)
		img_proceed = img_proceed/255.0
		img_proceed = img_proceed.unsqueeze(0)
		img_proceed = img_proceed.unsqueeze(0)

		return img_proceed

	def predict_expressions(self,img):
		"""
		predicting expressions
		"""
		img_proceed = self.model_eligible_format(img)
		attention, prediction = for_test(img_proceed)

		prediction_text = ""

		for i in range(attention.shape[0]):
			if prediction[i] == "<eol>":
				continue
			else:
				prediction_text += prediction[i]

		print(prediction_text)
		return (prediction_text)


	def run_for_std_scenario(self,img):
		# img_test = cv2.imread("./test_images/test12.png")
		# img_test = self.pre_process(img)
		EXPRESSIONS = Expressions(img)
		EXPRESSIONS.draw_contours()
		images = EXPRESSIONS.get_expressions()


		equations = []

		for image in images:
			cv2.imwrite(parent_dir+"\\results\\result_1.bmp",image)
			image = Image.open(parent_dir+"\\results\\result_1.bmp").convert("L")
			equations.append(self.predict_expressions(image))

		return equations

	def run_for_training_scenario(self,img):
		equations = self.predict_expressions(img)
		#close all the windows
		cv2.destroyAllWindows()
		if re.search(r'_\{.*?\}', equations):
			print("Pattern found, removing it.")
			# Use regex to find and remove all occurrences of _{something}
			cleaned_expression = re.sub(r'_\{.*?\}', '', equations)
			return cleaned_expression
		else:
			print("Pattern not found.")
			return equations
	
def convert_blackboard_image(img):
	img = cv2.erode(img, np.ones((2,2),np.uint8), iterations = 3)
	# cv2.imshow("img",img)
	# cv2.waitKey(1000)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY) # 85 # 155	
	# cv2.imshow("img",img)
	# cv2.waitKey(1000)

	kernel = np.ones((7,7),np.uint8)

	dilation = cv2.dilate(img, kernel, iterations = 8) #16

	contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	
	contours = [cnt for cnt in contours if (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3])>=0.5]

	#draw the largest contour
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
	x, y, w, h = cv2.boundingRect(contours[0])
	im2 = img.copy()
	rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 0)
	img = img[y:y+h, x:x+w]

	scale_factor = 0.4
	resized_image = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

	resized_image = cv2.dilate(resized_image, np.ones((4,4),np.uint8), iterations = 1)

	# cv2.imshow("img",im2)
	# cv2.waitKey(1000)

	cv2.imshow("img",resized_image)
	cv2.waitKey(1000)

	return resized_image


def convert_camera_image(img):
	img = cv2.dilate(img, np.ones((2,2),np.uint8), iterations = 1)
	# cv2.imshow("img",img)
	# cv2.waitKey(1000)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	kernel = np.ones((7,7),np.uint8)
	dilation = cv2.dilate(img, kernel, iterations = 8) #16

	contours,_ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	largest_contour = max(contours, key=cv2.contourArea)
	x, y, w, h = cv2.boundingRect(largest_contour)
	img = img[y:y+h, x:x+w]

	scale_factor = 0.4
	resized_image = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

	cv2.imshow("img",resized_image)
	cv2.waitKey(1000)

	return resized_image





def test1():
	img = cv2.imread(parent_dir+"./test_images/image.png")
	img = convert_camera_image(img)
	I2T = Image2Text()
	equations = I2T.run_for_training_scenario(img)
	print(equations)

def test2():
	img = cv2.imread(parent_dir+"./test_images/whiteboard.png")
	# cv2.imshow("img",img)
	# cv2.waitKey(1000)
	img2 = convert_blackboard_image(img)
	I2T = Image2Text()
	equations = I2T.run_for_training_scenario(img2)

if __name__ == "__main__":
	test2()