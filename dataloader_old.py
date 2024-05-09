import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def orb_function(master, new):
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
    kp_master, desc_master = orb.detectAndCompute(master, None)
    kp_new, desc_new = orb.detectAndCompute(new, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(desc_master, desc_new)
    except cv2.error as e:
        print("OpenCV error happened:", e)
        return master, 0
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches)==0:
        print("Error!!!")
        return new, 0
    else:
        good_matches=matches[0:20]
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

def data_loader(path, master, resize):
    '''
    path 를 입력으로 받음, stop, N=1000은 옵션. step = True로 하고 N개의 데이터만 콜렉팅함.
    이미지를 정규화해서 리턴.
    '''
    images = []

    n = 0
    for name in os.listdir(path):
        image_path = os.path.join(path, name)
        image = cv2.imread(image_path)
        image, switch = orb_function(master, image)
        if switch==0:
            pass
        else:
            if image.shape[0]==0 or image.shape[1]==0:
                pass
            else:
                if resize[0]:
                    # print(image.shape)
                    image = cv2.resize(image, (resize[1], resize[2])) # width, height
                images.append(image)
                if n % 100 == 0:
                    print("Loading ...", n)
                n += 1
    
    images_np = np.array(images)
    return images_np

def MS_loader(MS_list, master, resize = [False, 0, 0]):
    '''
    인풋으로 MS_list를 받음. 예시 :
    MS_list =  [['3029C005AA', 'step_1'], ['3029C006AA', 'step_1'], ['3029C009AA', 'step_1'], ['3029C010AA', 'step_1'],
                ['3030C002AA', 'step_1'], ['3030C003AA', 'step_1'], ['3030C004AA', 'step_1'], ['3031C001AA', 'step_1'],
                ['3031C002AA', 'step_1'], ['3031C003AA', 'step_1']]
    MS_list 안에 첫 번째 요소로 Mtype이 있으며 두 번째 요소로 step이 있음.
    '''
    for Mtype, step in MS_list:
        Mtype_path = os.path.join('../../images', Mtype) # '../../images' 주의할 것.
        load_path = os.path.join(Mtype_path, step) # '../../images' 주의할 것.
        name = os.listdir(load_path)[0]
        image_path = os.path.join(load_path, name)
        image = cv2.imread(image_path)
        if resize[0]:
            image = cv2.resize(image, (resize[1], resize[2]))
        height, width, channel = image.shape
        data = np.empty((0, height, width, channel))
        break
    for Mtype, step in MS_list:
        
        Mtype_path = os.path.join('../../images', Mtype) # '../../images' 주의할 것.
        load_path = os.path.join(Mtype_path, step) # '../../images' 주의할 것.
        Mtype_data = data_loader(load_path, master, resize)
        data = np.concatenate([data, Mtype_data], axis = 0)
    return data