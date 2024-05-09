import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

class loader:
    def __init__(self, master):
        self.master = master
    def orb_function(self, new, **kwargs):
        orb = cv2.ORB_create(**kwargs)
        kp_master, desc_master = orb.detectAndCompute(self.master, None)
        kp_new, desc_new = orb.detectAndCompute(new, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(desc_master, desc_new)
        except cv2.error as e:
            print("Error!!!", e)
            return None, 0
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches)==0:
            print("Error!!! Image is not matched at all.")
            return None, 0
        else:
            good_matches=matches[0:30]
            total_x = 0
            total_y = 0
            for match in good_matches:
                new_idx = match.trainIdx
                total_x += kp_new[new_idx].pt[0]
                total_y += kp_new[new_idx].pt[1]
            num_matches = len(good_matches)
            center_x = total_x / num_matches
            center_y = total_y / num_matches
            image = new[int(center_y) - 150:int(center_y) + 150, int(center_x) - 150:int(center_x) + 150, :]
            return image, 1

    def data_loader(self, n, path, resize):
        '''
        Input : 
        n      : counting loaded data variable
        path   : data path
        resize : [0] is an option for resizing. If [0] is set to true, the output image will be resized to ([1], [2]) (width, height).

        Output : n and images
        '''
        images = []
        
        for name in os.listdir(path):
            image_path = os.path.join(path, name)
            image = cv2.imread(image_path)
            image, switch = self.orb_function(image)
            if switch==0:
                pass
            else:
                if image.shape[0]==0 or image.shape[1]==0:
                    print("Error!!! Image has zero size.")
                    pass
                else:
                    if resize[0]:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (resize[1], resize[2])) # width, height
                images.append(image)
                n += 1
        
        images_np = np.array(images)
        return n, images_np

    def MS_loader(self, MS_list, resize = [False, None, None]):
        '''
        Input : MS_list and resize
        Output : Whole data chosen by MS_list
        '''
        for Mtype, step in MS_list:
            Mtype_path = os.path.join('../images', Mtype) # '../images' Warnings!!
            load_path = os.path.join(Mtype_path, step)
            name = os.listdir(load_path)[0]
            image_path = os.path.join(load_path, name)
            image = cv2.imread(image_path)
            if resize[0]:
                image = cv2.resize(image, (resize[1], resize[2]))
            height, width, channel = image.shape
            data = np.empty((0, height, width, channel))
            break
        n = 0 # Initialize #
        print("Loaded data : ", n)
        for Mtype, step in MS_list:
            
            Mtype_path = os.path.join('../images', Mtype) # '../images' Warnings!!
            load_path = os.path.join(Mtype_path, step)
            n, Mtype_data = self.data_loader(n, load_path, resize)
            data = np.concatenate([data, Mtype_data], axis = 0)
            print("Loaded data : ", n)
        return data

