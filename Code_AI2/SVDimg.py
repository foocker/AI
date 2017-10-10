import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import linalg as la

img = mpimg.imread('myheadportrait.jpg')
img_shape = img.shape
lum_img = img[:,:,0]  # get rid of the third dimension
imgplot = plt.imshow(img)
# plt.imshow(lum_img)
# plt.show()

def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def numsvdChoice():
    myMat = np.mat(lum_img)
    U, Sigma, VT = la.svd(myMat)
    Sigma2 = sorted(Sigma**2)  # small-->big
    threshold = sum(Sigma2) * 0.05
    for i in range(len(Sigma2)):
        if sum(Sigma2[:i+1]) >= threshold:
            numSVD = len(Sigma2) - i
            break
    return U, Sigma, VT, numSVD

def imgCompress(num_svd=numsvdChoice):
    U, Sigma, VT, numSVD = num_svd()
    SigRecon = np.mat(np.zeros((numSVD, numSVD)))
    print("numSVD:", numSVD)
    for k in range(numSVD):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:,:numSVD] * SigRecon * VT[:numSVD,:]
    return reconMat
print("Shape of the image{0} '\n' After SVD{1}:".format(lum_img.shape, imgCompress().shape))
print(imgCompress())
plt.imshow(imgCompress())
plt.show()
    
