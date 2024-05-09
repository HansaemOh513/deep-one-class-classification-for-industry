import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

class loader:
    def __init__(self, master):
        self.master = master
    def orb_function(self, new):
        orb = cv2.ORB_create()
        kp_master, desc_master = orb.detectAndCompute(self.master, None)
        kp_new, desc_new = orb.detectAndCompute(new, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc_master, desc_new)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches=matches[0:50]
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
        fig, ax = plt.subplots()
        ax.imshow(new)
        ax.scatter([center_x], [center_y], color='green', s=100)
        ax.text(center_x + 10, center_y + 10, 'Center', color='white', fontsize=12)
        plt.show()

        return image

    def data_loader(self, path, resize):
        '''
        path 를 입력으로 받음, stop, N=1000은 옵션. step = True로 하고 N개의 데이터만 콜렉팅함.
        이미지를 정규화해서 리턴.
        '''
        images = []
        
        n = 0
        for name in os.listdir(path):
            image_path = os.path.join(path, name)
            image = cv2.imread(image_path)
            image = self.orb_function(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if resize[1]:
                image = cv2.resize(image, (resize[2], resize[3])) # width, height
            images.append(image/255)
            if n % 100 == 0:
                print("Loading ...", n)
            n += 1
        
        images_np = np.array(images)
        
        return images_np

    def MS_loader(self, MS_list, resize = [False, False, 0, 0]):
        '''
        인풋으로 MS_list를 받음. 예시 :
        MS_list =  [['3029C005AA', 'step_1'], ['3029C006AA', 'step_1'], ['3029C009AA', 'step_1'], ['3029C010AA', 'step_1'],
                    ['3030C002AA', 'step_1'], ['3030C003AA', 'step_1'], ['3030C004AA', 'step_1'], ['3031C001AA', 'step_1'],
                    ['3031C002AA', 'step_1'], ['3031C003AA', 'step_1']]
        MS_list 안에 첫 번째 요소로 Mtype이 있으며 두 번째 요소로 step이 있음.
        '''
        def data_shape():
            for Mtype, step in MS_list:
                Mtype_path = os.path.join('../../images', Mtype) # '../../images' 주의할 것.
                load_path = os.path.join(Mtype_path, step) # '../../images' 주의할 것.
                Mtype_data = self.data_loader(load_path, resize)
                break
            _, height, width, channel =  Mtype_data.shape
            return (0, height, width, channel)
        data = np.empty(data_shape())
        for Mtype, step in MS_list:
            
            Mtype_path = os.path.join('../../images', Mtype) # '../../images' 주의할 것.
            load_path = os.path.join(Mtype_path, step) # '../../images' 주의할 것.
            Mtype_data = self.data_loader(load_path, resize)
            data = np.concatenate([data, Mtype_data], axis = 0)
        return data

