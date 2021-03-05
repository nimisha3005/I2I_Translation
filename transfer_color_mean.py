# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import skimage as sk
import skimage.io as skio
from matplotlib import pyplot as plt
from PIL import Image 
from math import *
  
def cal_mean(L,A,B):
    """
    Calculating mean of each channel
    """
    
    L_mean = L.mean()
    A_mean = A.mean()
    B_mean = B.mean()
    return L_mean,A_mean,B_mean

def cal_std(L,A,B):
    """
    Calculating standard deviation of each channel
    """
    
    L_std = L.std()
    A_std = A.std()
    B_std = B.std()
    return L_std,A_std,B_std

def PSNR(original, target): 
    """
    Calculate Peak Signal to Noise Ratio:
        Greater value indicates better quality of reconstructed image
    """
    original = cv2.resize(original,(target.shape[1],target.shape[0]))
    mse = np.mean((original - target) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def split_channel(ref,tar):
    """
    Splitting reference and target images to LAB values 
    """
    
    RL,RA,RB=cv2.split(ref)
    TL,TA,TB=cv2.split(tar)
    
    #Calculating mean of reference and target images
    RL_mean,RA_mean,RB_mean=cal_mean(RL,RA,RB)
    TL_mean,TA_mean,TB_mean=cal_mean(TL,TA,TB)
    
    #Calculating standard deviation of reference and target images
    RL_std,RA_std,RB_std=cal_std(RL,RA,RB)
    TL_std,TA_std,TB_std=cal_std(TL,TA,TB)
        
    TL = TL.astype('float')
    TA = TA.astype('float')
    TB = TB.astype('float')
    
    h,w,c = tar.shape
    r_mean = ([RL_mean,RA_mean,RB_mean])
    t_mean = ([TL_mean,TA_mean,TB_mean])
    r_std = ([RL_std,RA_std,RB_std])
    t_std = ([TL_std,TA_std,TB_std])
    
    #Apply color correction to each pixel of target image
    for i in range(h):
        for j in range(w):
            for k in range(c):
                pixel = tar[i,j,k]
                pixel = round(((pixel-t_mean[k])*(r_std[k]/t_std[k]))+r_mean[k])
                
                if pixel<0:
                    pixel=0
                if pixel>255:
                    pixel=255
                    
                tar[i,j,k] = pixel
                
    
    result = np.array(tar, dtype=np.uint8)
    result = cv2.cvtColor(result,cv2.COLOR_LAB2BGR)
    #result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
    
    return result

def white_balance(img):
    """
    Function to perform white balancing on the input image
    """

    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def auto_contrast(img):
    """
    Function to perform auto contrast on the input image
    """

    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(result)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def read_img(reference, target):
    """
    Reading images and setting the area of reference image to use
    """
    
    r = skio.imread(reference)
    
    #Uncomment to use this code for setting area of reference image to use
    """
    pil_image = Image.fromarray(r)
    pil_image = pil_image.crop((400, 100, 800, 300)) #left,top,right,bottom
    cv_image = np.array(pil_image) 
    r = cv_image
    """
    
    t = skio.imread(target)
    
    skio.imshow(r)
    plt.title("Reference Image")
    plt.show()
    
    skio.imshow(t)
    plt.title("Target Image")
    plt.show()
    
    
    r = cv2.cvtColor(r,cv2.COLOR_BGR2LAB)
    t = cv2.cvtColor(t,cv2.COLOR_BGR2LAB)
    
    
    
    return r, t

def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std


ref,tar = read_img(r"Source Image Path",r"Target Image Path")
res = split_channel(ref,tar)

skio.imshow(res)
plt.title("Resultant Image")
plt.show()

value = PSNR(ref,res)

wb = white_balance(res)
skio.imshow(wb)
plt.title("WB")
plt.show()

ac = auto_contrast(wb)
skio.imshow(ac)
plt.title("AC")
plt.show()
