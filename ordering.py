import os
import shutil
import glob
import cv2
import sys
import re
destination_path = "./images"
def move():
    path_1 = "./data" # Top level folder containing the data
    folder_1 = os.listdir(path_1) # Machine types
    for label in folder_1:
        path_2 = os.path.join(path_1, label) 

        folder_2 = os.listdir(path_2) # Machine information
        for printer in folder_2:
            path_3 = os.path.join(path_2, printer)
            folder_3 = os.listdir(path_3) # NG or OK information
            for name in folder_3 :
                path_4 = os.path.join(path_3, name) # Image path
                if re.search("jpg", name):
                    if re.search(pattern, name):
                        print(destination_path, label, pattern_)
                        
                        pattern__ = re.sub(".jpg", "", pattern_)
                        os.makedirs(os.path.join(destination_path, label, pattern__), exist_ok=True)
                        shutil.copy(path_4, os.path.join(destination_path, label, pattern__, printer + name))

for i in range(30):
    pattern = f'step_{i}.jpg'
    pattern_ = f'step_{i}.jpg'
    move()


