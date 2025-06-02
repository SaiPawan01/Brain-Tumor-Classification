import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import random
import numpy as np

dataset_dir = "Dataset"
test_dir = "Training"

test_data_dir = Path(dataset_dir+'/'+test_dir)
print(test_data_dir)

# walk throught directories
print("===========Directory Structure=========")
for dirpath,dirnames,filename in os.walk(test_data_dir):
        print(f"there are {len(dirnames)} directories and {len(filename)} images in '{dirpath}")


#making list of images
test_data_imgs_list = list(test_data_dir.glob('*/*.jpg'))
# print(test_data_imgs_list)


# ploting random images
plt.figure(figsize=(12,12))
for i in range(12):
        plt.subplot(3,4,i+1)
        image_path = random.choice(test_data_imgs_list)
        label_dir = os.path.dirname(image_path)
        _,label = os.path.split(label_dir)
        # print(label)
        image = np.asarray(Image.open(image_path))
        plt.imshow(image)
        plt.axis('off')
        plt.title(label)
        plt.grid()
        
plt.show()
        