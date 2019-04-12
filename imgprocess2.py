from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#train = "./my-dataset/single/train/"
test = "./my-dataset/single/test/"

def loadImages(cls): 
    classes = [f for f in os.listdir(test)]
    print("Working with {0} classes".format(len(classes)))
    print("Working with {0} class".format(cls))
    files = [f for f in os.listdir(test + cls + '/') if os.path.isfile(os.path.join(test + cls + '/', f))]
    return files

def saveImage(X, cls, i):
    #plt.figure(figsize=(1.5, 1.5))
    #plt.imshow(X.reshape(100,100,3))
    #plt.show()
    #plt.close()
    if not os.path.isdir("./pre-dataset/"):
        os.mkdir("./pre-dataset/")
    if not os.path.isdir("./pre-dataset/single/"):
        os.mkdir("./pre-dataset/single/")
    if not os.path.isdir("./pre-dataset/single/test/"):
        os.mkdir("./pre-dataset/single/test/")
    if not os.path.isdir("./pre-dataset/single/test/" + cls + "/"):
        os.mkdir("./pre-dataset/single/test/" + cls + "/")
    
    new_img = array_to_img(X.reshape(100,100,3))
    new_img.save("./pre-dataset/single/test/"+ cls + "/" + str(i) + ".jpg")

def buildDataset(files, cls):
    image_width = 100
    image_height = 100
    channels = 3
    dataset = np.ndarray(shape=(len(files), image_height, image_width, channels),dtype=np.float32)
    i = 0
    for f in files:
        img = cv2.imread(test + cls + "/" + f)
        img = cv2.resize(img,(100,100))
        b,g,r = cv2.split(img)
        rgb_img = cv2.merge([r,g,b])
        x = img_to_array(rgb_img)
        dataset[i] = rgb_img
        i += 1
    print("All images to array!")
    return dataset

def whitenImages(X):
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
    print('new shape ', X.shape)

    X_norm = X / 255.
    print('X.min()', X_norm.min())
    print('X.max()', X_norm.max())

    X_norm = X_norm - X_norm.mean(axis=0)
    print('Norm: ', X_norm)
    cov = np.cov(X_norm, rowvar=True)
    U,S,V = np.linalg.svd(cov)
    epsilon = 0.1
    X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X_norm)
    X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())
    return X_ZCA_rescaled

def preprocess():
    classes = [f for f in os.listdir(test)]
    for cls in classes:
        files = [f for f in os.listdir(test + cls + '/') if os.path.isfile(os.path.join(test + cls + '/', f))]
        dataset = whitenImages(buildDataset(files, cls))
        for i in range(len(dataset)):    
            saveImage(dataset[i,:], cls, i)
    
preprocess()

