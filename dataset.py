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
    x_y = []

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
            x_y.append((pixel_data, y_i))
            count += 1 
            
            if count == selected_images:
                break

    return x_y

def shuffle_dataset(x_y):
    random.shuffle(x_y)
    x = []
    y = []

    for i, j in x_y:
        x.append(i)
        y.append(j)

    return x, y