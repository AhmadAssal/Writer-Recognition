import numpy as np
import os
import cv2 as cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
from sklearn.svm import SVR
from skimage import feature
from tqdm import tqdm
from sklearn.svm import LinearSVC
import random
from datetime import datetime
import shutil
import glob,fnmatch




def show(img : numpy dtype object, factor = 1.0 : float, name = "image" : str) -> None:
    """Show an image until the escape key is pressed
    If the arguments factor or name are not given the default : 1, "name" are used respectively 
    Parameters
    ----------
    img : numpy dtype object
    factor : float , optional
    name : str , optional 
    """
    if factor != 1.0:
        img = cv2.resize(img, (0,0), fx=factor, fy=factor) 

    cv2.imshow(name,img)
    while(1):
        k = cv2.waitKey(0)
        if k==27:    # Esc key to stop
            break
    cv2.destroyAllWindows()



def read_image(input_path :  str) -> numpy dtype :
    """Reads an image and turns it into gray scale and returns it
    Parameters
    ----------
    input_path : str
    """ 
    img = cv2.imread(input_path,0)
return img


def sort_lists(list1 : list , list2 : list) -> list:
    """Sorts list2 based on list1 where list1 is a list of numbers
    Parameters
    ----------
    list1 : list
    list2 : list
    """ 
    list1 = np.asarray(list1)
    sorter = np.argsort(list1)
    list3 = []
    for i in range(len(sorter)):
        list3.append(list2[sorter[i]])
    return list3

def read_directory(input_path):
    cases = []
    for file in os.listdir(input_path):
        cases.append(int(file))
    cases.sort()
    return cases