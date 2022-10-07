import numpy as np
import cv2 
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from math import exp,sqrt


def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def FPBG(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def FPAG(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def main():
    img = cv2.imread('Image/lena_gray_512.tif', cv2.IMREAD_GRAYSCALE)
    npImg = np.array(img)

    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    
    imgPassBaixa = FPBG(50,img.shape)
    imgProcessBaixa = imgTrans*imgPassBaixa

    inversaBaixa = np.fft.ifftshift(imgProcessBaixa) 
    inversaBaixa = np.fft.ifft2(inversaBaixa)  
    inversaBaixa = np.abs(inversaBaixa)

    imgPassAlta = FPAG(50,img.shape)
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