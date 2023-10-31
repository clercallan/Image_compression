from PIL import Image
from numpy import asarray
import numpy as np
import math

im_name = "pns_original.jpeg"

#Ouverture de l'image
image = Image.open(im_name)

#Rogner l'image
croppedImage = image.crop((0, 0, image.width- (image.width % 8), image.height - (image.height % 8)))

# separre en 3 matrices 2D
r, g, b = croppedImage.split()

debut = np.asarray(croppedImage)

# converti les images en array et centrage des valeurs
decalage = np.full(np.asarray(r).shape, 128)
matR = np.asarray(r) - decalage
matG = np.asarray(g) - decalage
matB = np.asarray(b) - decalage


## calcul des coef de la matrice P
P = np.zeros((8, 8))
for k in range(8):
    for n in range(8):
        if k == 0:
            P[k, n] = np.sqrt(1 / 8)
        else:
            P[k, n] = np.sqrt(2 / 8) * np.cos((np.pi * k * (1 / 2 + n)) / 8)

            
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 13, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])


matrices = [matR,matG,matB]
matricesComp = []
nonZeros = 0

for m in matrices :

    ######### Compression ##########

    F = 2 #frequence de coupure

    matComp = np.zeros(m.shape) #matrice compressee finale

    nbLignes = len(m)//8
    nbColonnes = len(m[1])//8
    for i in range (nbLignes) :
        for j in range (nbColonnes) :
            M = m[i*8:(i+1)*8,j*8:(j+1)*8]
            
            D = P @ M @ np.transpose(P)
            comp = np.trunc(D/Q) # 8x8 compressed bloc

            # remove high frequences 
            for l in range (8) :
                for k in range (8) :
                    if l + k >= F:
                        comp[l,k] = 0
                    
                        
            matComp[i*8:(i+1)*8,j*8:(j+1)*8] = comp #puts the bloc in the finale matrix

    matricesComp.append(matComp)
    nonZeros += np.count_nonzero(matComp)

TP = P.transpose()
decFinale=[]

##Décompression
for m in matricesComp:
    matDec = np.zeros(m.shape)
    for i in range(m.shape[0] // 8):
        for j in range(m.shape[1] // 8):
            M = m[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]*Q
            DM = TP @ M @ P
            for l in range(8):
                for k in range(8):
                    if DM[l, k] > 127:
                        DM[l, k] = 127
                    if DM[l, k] < -128:
                        DM[l, k] = -128
            matDec[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = (DM + np.full((8,8), 128)).astype(int)
    decFinale.append(matDec)


finito = np.dstack((decFinale[0], decFinale[1], decFinale[2])).astype(np.uint8)


##### NORME #######

print("Taux de compression : ",(1200*600*3 - nonZeros)/(12*600*3))

print("Norme L2 :", np.linalg.norm(debut-finito)/(12*600*3))

img = Image.open(im_name)
x = np.asarray(img)
Image.fromarray(finito).show()
Image.fromarray(finito).save('img_compressed.jpeg')






