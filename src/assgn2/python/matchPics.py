import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2):
	#I1, I2 : Images to match
	

	#Convert Images to GrayScale
	
	
	#Detect Features in Both Images
	
	
	#Obtain descriptors for the computed feature locations
	

	#Match features using the descriptors
	

	return matches, locs1, locs2
