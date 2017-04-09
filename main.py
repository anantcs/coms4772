from process_image import read, show
from sparsenet import sparsenet

import numpy as np

if __name__ == '__main__':
    data = read()
    images_array = []
    for label, image in data:
        images_array.append(image)
    image_np_array = np.array(images_array)
    print(image_np_array.shape)
    image_np_array_reshaped = np.reshape(image_np_array, (60000, 784, 1))
    print(image_np_array_reshaped.shape)

    phi = sparsenet(image_np_array)
    print(phi)
