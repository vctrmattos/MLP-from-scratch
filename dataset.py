import os
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

def read_mnist(mode:str, p: float):
    dataset_folder = os.path.join("./dataset/mnist_png", mode)
    x = []
    y = []
    
    for i in range(2):
        dir = os.path.join(dataset_folder, str(i))
        print(dir)
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
            x.append(pixel_data)
            y.append(i)
            count += 1 
            if count == selected_images:
                break
    return x, y

x, y = read_mnist("testing", 0.01)
