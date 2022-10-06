import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft, ifft
from math import sqrt


def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def FPBB(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def FPAB(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def main():
    img = cv2.imread('img/lena_gray_512.tif', cv2.IMREAD_GRAYSCALE)
    npImg = np.array(img)

    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    
    imgPassBaixa = FPBB(50,img.shape, 20)
    imgProcessBaixa = imgTrans*imgPassBaixa

    inversaBaixa = np.fft.ifftshift(imgProcessBaixa) 
    inversaBaixa = np.fft.ifft2(inversaBaixa)  
    inversaBaixa = np.abs(inversaBaixa)

    imgPassAlta = FPAB(50,img.shape, 20)
    imgProcessAlta = imgTrans*imgPassAlta
    inversaAlta = np.fft.ifftshift(imgProcessAlta) 
    inversaAlta = np.fft.ifft2(inversaAlta)  
    inversaAlta = np.abs(inversaAlta)

    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax[0,0].imshow(img, cmap='gray')
    ax[0,0].set_title("Original")
    ax[0,1].imshow(imgPassBaixa, cmap='gray')
    ax[0,1].set_title("Imagem Filtro Pass. Baixa")
    ax[0,2].imshow(inversaBaixa, cmap='gray')
    ax[0,2].set_title("Imagem Resultante")
    
    ax[1,0].imshow(img, cmap='gray')
    ax[1,0].set_title("Original")
    ax[1,1].imshow(imgPassAlta, cmap='gray')
    ax[1,1].set_title("Imagem Filtro Pass. Alta")
    ax[1,2].imshow(inversaAlta, cmap='gray')
    ax[1,2].set_title("Imagem Resultante")
    plt.show() 
    
if __name__ == "__main__":
    main()