import os
import random
import tarfile
import requests
from PIL import Image
from linearAlgebra import Matrix

def get_mnist():
    url = "https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"
    filename = "mnist_png.tar.gz"

    req = requests.get(url)

    with open(filename, "wb") as f:
        f.write(req.content)

    file = tarfile.open(filename)
    file.extractall('./dataset')
    file.close()



def read_dataset(mode:str, p: float):
    
    dataset_folder = os.path.join("./dataset/mnist_png", mode)
    xy = []

    for y_i in range(10):
        dir = os.path.join(dataset_folder, str(y_i))
        x_size_i = len(os.listdir(dir))
        count = 0
        selected_images = round(p*x_size_i)
        
        for j in os.listdir(dir):
            image = Image.open(os.path.join(dir, j))
            image = image.convert('L')
            width, height = image.size
            pixel_data = []
            
            for h in range(height):
                row_data = []
                for w in range(width):
                    pixel_value = image.getpixel((w, h))
                    row_data.append(pixel_value)
                pixel_data.append(row_data)
            xy.append((Matrix(pixel_data), y_i))
            count += 1 
            
            if count == selected_images:
                break

    return xy

def shuffle_dataset(xy):
    random.shuffle(xy)
    x = []
    y = []

    for i, j in xy:
        x.append(i)
        y.append(j)

    return x, y

def to_categorical(y):
    categorical = Matrix.zeros((len(y), 10))
    for i in range(len(y)):
        categorical[i, y[i]] = 1

    return categorical

def mnist_acc(y_test:Matrix, predictions:list): 
    cont = 0
    for i in range(len(predictions)):
        if predictions[i].index(max(predictions[i])) == y_test[i].array[0].index(max(y_test[i].array[0])):
            cont += 1
    return cont/len(predictions)