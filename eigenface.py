import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


def resize(image, min_row, min_col):
    row, col = image.shape
    top, bot, left, right = 0, row, 0, col
    if row > min_row:
        top = row - min_row
    if col > min_col:
        right = min_col
    return image[top:bot, left:right]


def getweights(newim, eigenface):
    # newim = np.subtract(newim, meanimage)
    a = []
    newim = np.transpose(newim)
    for item in range(len(eigenface)):
        ai = np.dot(newim, eigenface[item])
        a.append(ai)
    return a


def create_new_image(eigenface, weights):
    newface = np.zeros(eigenface[0].shape, dtype=np.uint8)
    for i in range(len(eigenface)):
        newface0 = np.multiply(eigenface[i], weights[i])
        newface = np.add(newface, newface0)
    # newface = np.add(newface, meanimage)
    return newface


min_rows, min_cols = sys.maxsize, sys.maxsize
path = "dataset/"
images = os.listdir(path)

g_images = []

for im in images:
    image = cv2.imread(path + im, flags=0)
    g_images.append(image)

for (i, image) in enumerate(g_images):
    row, col = image.shape[0], image.shape[1]
    min_rows = min(min_rows, row)
    min_cols = min(min_cols, col)

if min_cols < min_rows:
    min_size = min_cols
else:
    min_size = min_rows

final_images = []

for im in g_images:
    resized = resize(im, min_row=min_size, min_col=min_size)
    final_images.append(resized)

size = final_images[0].shape

data = np.ndarray(shape=(len(final_images), size[0] * size[1]), dtype=np.uint8)

for i in range(len(final_images)):
    image = np.array(final_images[i], dtype=np.uint8).flatten()
    data[i, :] = image

mean, eigenvector = cv2.PCACompute(data, mean=None, maxComponents=100)

mean = mean.astype(np.uint8)
meanimage = mean.reshape(size)

eigenfaces = []
for vector in eigenvector:
    face = vector.reshape(size)
    eigenfaces.append(face)

plot1 = plt.figure(1)
plt.imshow(meanimage, cmap="gray")

# print(eigenvector[0].shape)
plot2 = plt.figure(2)
for item in range(10):
    plt.subplot(3, 5, item + 1)
    plt.imshow(eigenfaces[item], cmap="gray")


new_face = cv2.imread("test-set/000032.jpg", flags=0)

new_face = resize(new_face, min_row=min_size, min_col=min_size)
new_flat_face = np.array(new_face, dtype=np.uint8).flatten()
# print(new_flat_face.shape)

weights = getweights(new_flat_face, eigenvector)
created_image = create_new_image(eigenfaces, weights)

plot3 = plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(new_face, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(created_image, cmap="gray")
plt.show()

# print(eigenfaces[0].max())
