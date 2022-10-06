import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft, ifft
from math import sqrt


def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def PassaBanda(D0,D1,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0

    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D1:
                base[y,x] = 1
    return base

def main():
    img = cv2.imread('Imagem/jornal.tif', cv2.IMREAD_GRAYSCALE)
    npImg = np.array(img)

    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))

    ImgPassBanda = PassaBanda(250,50,img.shape)
    imgProcessBanda = ImgPassBanda*imgTrans

    inversaBanda = np.fft.ifftshift(imgProcessBanda) 
    inversaBanda = np.fft.ifft2(inversaBanda)  
    inversaBanda = np.abs(inversaBanda)
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original")
    ax[1].imshow(ImgPassBanda, cmap='gray')
    ax[1].set_title("Imagem filtro")
    ax[2].imshow(inversaBanda, cmap='gray')
    ax[2].set_title("Imagem Resultante")
    plt.show() 

if __name__ == "__main__":
    main()