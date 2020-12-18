import cv2 as cv
import numpy as np
from RGB2Lab import rgb2lab
from utils import *


def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    # -----fill in your implementation here --------


    # ----------------------------------------------
    
    return filterResponses
