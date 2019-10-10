import os  
from PIL import Image  
import csv
import numpy as np  
from keras import backend as K
import scipy.io

def load_sample(): #ct window
     
    img = np.load("./data/sample_64.npy") #Deeplearningallpatchsmalls  3part3
    datatrain = np.asarray(img,dtype="float32")


    return datatrain
