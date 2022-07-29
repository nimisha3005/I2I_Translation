# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:21:53 2021

@author: Dell
"""

from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
import numpy as np
import skimage as sk
import skimage.io as skio
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image,ImageFilter

def pdf_1d(arr_in, arr_ref):
        """
        Apply 1-dim probability density function (PDF) transfer.
        """

        arr = np.concatenate((arr_in, arr_ref))
        # discretization as histogram
        a_min = arr.min() 
        a_max = arr.max() 
        pt = np.array([a_min + (a_max - a_min) * i / 300 for i in range(300 + 1)])
        hist_in, b_in = np.histogram(arr_in, pt)
        hist_ref, _ = np.histogram(arr_ref, pt)
         
        bin_centers = 0.5*(pt[1:] + pt[:-1])
        import scipy.stats as stats
        
        # Compute the PDF on the bin centers from scipy distribution object
        pdf = stats.norm.pdf(bin_centers)
        #plt.plot(pdf, label="pdf")
        #plt.plot(hist_ref, label="Histogram of ref")
        #plt.legend()
        #plt.show()
        
        
        pt = pt[:-1]
        # compute probability distribution
        cum_in = np.cumsum(hist_in) #Cumulative sum of source image histogram
        cum_ref = np.cumsum(hist_ref)   #Cumulative sum of target image histogram
        
        ratio_in = cum_in / cum_in[-1]
        ratio_ref = cum_ref / cum_ref[-1]
        
        # transfer pixel values
        t_d_in = np.interp(ratio_in, ratio_ref, pt)
        t_d_in[ratio_in <= ratio_ref[0]] = a_min
        t_d_in[ratio_in >= ratio_ref[-1]] = a_max
        
        arr_out = np.interp(arr_in, pt, t_d_in)
        return arr_out

def pdf_transfer_nd(arr_in=None, arr_ref=None, step_size=1):
        """
        Apply n-dim probability density function transfer.
        """
        #The values for rotation matrices have been taken from original MATLAB implementation of the paper
        rotation_matrices = [
            [
                [1.000000, 0.000000, 0.000000],
                [0.000000, 1.000000, 0.000000],
                [0.000000, 0.000000, 1.000000],
            ],
            [
                [0.333333, 0.666667, 0.666667],
                [0.666667, 0.333333, -0.666667],
                [-0.666667, 0.666667, -0.333333],
            ],
            [
                [0.577350, 0.211297, 0.788682],
                [-0.577350, 0.788668, 0.211352],
                [0.577350, 0.577370, -0.577330],
            ],
            [
                [0.577350, 0.408273, 0.707092],
                [-0.577350, -0.408224, 0.707121],
                [0.577350, -0.816497, 0.000029],
            ],
            [
                [0.332572, 0.910758, 0.244778],
                [-0.910887, 0.242977, 0.333536],
                [-0.244295, 0.333890, -0.910405],
            ],
            [
                [0.243799, 0.910726, 0.333376],
                [0.910699, -0.333174, 0.244177],
                [-0.333450, -0.244075, 0.910625],
            ],
        ]
        
        rotation_matrices = [np.array(x) for x in rotation_matrices]
        
        # n times of 1d-pdf-transfer
        arr_out = np.array(arr_in)
        for rotation_matrix in rotation_matrices:
            rot_1 = np.matmul(rotation_matrix, arr_out)
            #plt.plot(rot_arr_in, label="Img",color="orange")
            #plt.show()        
        
            rot_2 = np.matmul(rotation_matrix, arr_ref)
            rot_arr_out = np.zeros(rot_1.shape)
            for i in range(rot_arr_out.shape[0]):
                rot_arr_out[i] = pdf_1d(rot_1[i], rot_2[i])
            rot_delta_arr = rot_arr_out - rot_1
            delta_arr = np.matmul(
                rotation_matrix.transpose(), rot_delta_arr
            )
            arr_out = step_size * delta_arr + arr_out
        return arr_out

class Regrain:
    def __init__(self, smoothness=1):
        """To understand the meaning of these params, refer to paper07."""
        self.nbits = [4,16, 32, 64, 64,64]
        self.smoothness = smoothness
        self.level = 0
        self.l=[]

    def regrain(self, img_arr_in=None, img_arr_col=None):
        """keep gradient of img_arr_in and color of img_arr_col. """

        img_arr_in = img_arr_in / 255.0
        img_arr_col = img_arr_col / 255.0
        img_arr_out = np.array(img_arr_in)
        img_arr_out = self.regrain_rec(
            img_arr_out, img_arr_in, img_arr_col, self.nbits, self.level
        )
        img_arr_out[img_arr_out < 0] = 0
        img_arr_out[img_arr_out > 1] = 1
        img_arr_out = (255.0 * img_arr_out).astype("uint8")
        return img_arr_out

    def regrain_rec(self, img_arr_out, img_arr_in, img_arr_col, nbits, level):
        """direct translation of matlab code. """

        [h, w, _] = img_arr_in.shape
        print(img_arr_out.shape,img_arr_in.shape)
        h2 = (h + 1) // 2
        w2 = (w + 1) // 2
        if len(nbits) > 1 and h2 > 20 and w2 > 20:
            resize_arr_in = cv2.resize(
                img_arr_in, (w2, h2), interpolation=cv2.INTER_LINEAR
            )
            resize_arr_col = cv2.resize(
                img_arr_col, (w2, h2), interpolation=cv2.INTER_LINEAR
            )
            resize_arr_out = cv2.resize(
                img_arr_out, (w2, h2), interpolation=cv2.INTER_LINEAR
            )
            
            print("in",resize_arr_in.shape,resize_arr_out.shape)
            resize_arr_out = self.regrain_rec(
                resize_arr_out, resize_arr_in, resize_arr_col, nbits[1:], level + 1
            )
            skio.imshow(resize_arr_out)
            plt.title("Out"+str(nbits))
            plt.show()

            img_arr_out = cv2.resize(
                resize_arr_out, (w, h), interpolation=cv2.INTER_LINEAR
            )
            skio.imshow(img_arr_out)
            plt.title("In"+str(nbits))
            plt.show()
            
        skio.imshow(img_arr_out)
        plt.title("Out"+str(nbits))
        plt.show()
        skio.imshow(img_arr_in)
        plt.title("Out"+str(nbits))
        plt.show()

        img_arr_out = self.solve(img_arr_out, img_arr_in, img_arr_col, nbits[0], level)
        skio.imshow(img_arr_out)
        plt.title("Res")
        plt.show()

            #self.l.append(img_arr_out)
        return img_arr_out
    
   
    def helper_func(self, img_arr_out, img_arr_in, img_arr_col, nbit, level, eps=1e-6):
        """direct translation of matlab code. """

        [width, height, c] = img_arr_in.shape
        first_pad_0 = lambda arr: np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)
        first_pad_1 = lambda arr: np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
        last_pad_0 = lambda arr: np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
        last_pad_1 = lambda arr: np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)
       
        delta_x = last_pad_1(img_arr_in) - first_pad_1(img_arr_in)
        delta_y = last_pad_0(img_arr_in) - first_pad_0(img_arr_in)
        delta = np.sqrt((delta_x ** 2 + delta_y ** 2).sum(axis=2, keepdims=True))

        psi = 256 * delta / 5
        psi[psi > 1] = 1
        phi = 30 * 2 ** (-level) / (1 + 10 * delta / self.smoothness)

        phi1 = (last_pad_1(phi) + phi) / 2
        phi2 = (last_pad_0(phi) + phi) / 2
        phi3 = (first_pad_1(phi) + phi) / 2
        phi4 = (first_pad_0(phi) + phi) / 2

        rho = 1 / 5.0
        for i in range(nbit):
            den = psi + phi1 + phi2 + phi3 + phi4
            #print(phi3.shape)
            num = (
                np.tile(psi, [1, 1, c]) * img_arr_col
                + np.tile(phi1, [1, 1, c])
                * (last_pad_1(img_arr_out) - last_pad_1(img_arr_in) + img_arr_in)
                + np.tile(phi2, [1, 1, c])
                * (last_pad_0(img_arr_out) - last_pad_0(img_arr_in) + img_arr_in)
                + np.tile(phi3, [1, 1, c])
                * (first_pad_1(img_arr_out) - first_pad_1(img_arr_in) + img_arr_in)
                + np.tile(phi4, [1, 1, c])
                * (first_pad_0(img_arr_out) - first_pad_0(img_arr_in) + img_arr_in)
            )
            img_arr_out = (
                num / np.tile(den + eps, [1, 1, c]) * (1 - rho) + rho * img_arr_out
            )
        return img_arr_out

ref,tar = (r"F:\ML\Image_Color\cherry.jpg",r"F:\ML\Image_Color\tree.jpg")
r = skio.imread(ref)
t = skio.imread(tar)

[h, w, c] = t.shape
reshape_arr_in = t.reshape(-1, c).transpose() / 255.0
reshape_arr_ref = r.reshape(-1, c).transpose() / 255.0

a = pdf_transfer_nd(reshape_arr_in,reshape_arr_ref)
a[a < 0] = 0
a[a > 1] = 1
a = (255.0 * a).astype("uint8")
img_arr_out = a.transpose().reshape(h, w, c)
#print(type(reshape_arr_in),type())
skio.imshow(r)
plt.title("Reference")
plt.show()

skio.imshow(t)
plt.title("Target")
plt.show()


skio.imshow(img_arr_out)
plt.title("Result")
plt.show()

save_path = "F:\ML\Image_Color\Transfer_Color"
r=Regrain()

o = r.regrain(t,img_arr_out)
skio.imshow(o)
plt.title("New")
plt.show()
cv2.imwrite(os.path.join(save_path ,r'transfer_pdf'+'.jpg'), cv2.cvtColor(o, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)



result = cv2.bilateralFilter(o,19,75,75)
skio.imshow(result)
plt.title("Smooth")
plt.show()
"""
gray_o = cv2.cvtColor(img_arr_out, cv2.COLOR_BGR2GRAY)
gray_s = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
detail = gray_o - gray_s
final = gray_o + detail
f = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)"""
"""
pil_image = Image.fromarray(img_arr_out)
sharpened1 = pil_image.filter(ImageFilter.SHARPEN);
sharpened2 = sharpened1.filter(ImageFilter.SHARPEN);

skio.imshow(np.array(sharpened1))
plt.title("Final")
plt.show()
skio.imshow(np.array(sharpened2))
plt.title("Final")
plt.show()
"""
"""for i in o:

    skio.imshow(o)
    plt.title("New")
    plt.show()
"""
"""
out=np.array(reshape_arr_in)
con = np.concatenate((reshape_arr_in[0],reshape_arr_ref[0]))
min_arr = con.min()
max_arr = con.max()
xs = np.array([min_arr + (max_arr - min_arr) * i / 300 for i in range(300 + 1)])
print()
hist = np.histogram(reshape_arr_in[0].transpose(),xs)
print(reshape_arr_in[0].shape)
bin_centers = 0.5*(xs[1:] + xs[:-1])
import scipy.stats as stats
# Compute the PDF on the bin centers from scipy distribution object
pdf = stats.norm.pdf(bin_centers)

#print(reshape_arr_in[0].shape)
print(np.cumsum(hist))
print(np.cumsum(pdf))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(bin_centers, hist, label="Histogram of samples")
plt.plot(bin_centers, pdf, label="PDF")
plt.legend()
plt.show()
"""

