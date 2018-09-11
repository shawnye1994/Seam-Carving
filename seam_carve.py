# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:29:56 2018

@author: ShawnYe
"""

import cv2
import numpy as np
import time

def path_search_and_crop(original_img):
    """
    function of find and crop the lowest energy path along the vertical direction
    """
    original_shape = original_img.shape
    #initial the returned img which was cropped the lowest energy seam of original image
    if len(original_shape) == 3:
        single_crop = np.zeros((original_shape[0], original_shape[1]-1, 3), dtype = original_img.dtype)
    else:
        single_crop = np.zeros((original_shape[0], original_shape[1]-1), dtype = original_img.dtype)
    #calculate the energy map of original image by Laplacian operator       
    img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    energy_map = cv2.Laplacian(img_gray, cv2.CV_64F)
    energy_map = np.absolute(energy_map)
    accumulated_map = energy_map
    H,W = energy_map.shape
    for h in range(1,H):
        for w in range(0,W):
            if w == 0:
                upper_pixels = [energy_map[h-1,w], energy_map[h-1, w+1]]
            elif w == W-1:
                upper_pixels = [energy_map[h-1, w-1], energy_map[h-1,w]]
            else:
                upper_pixels = [energy_map[h-1, w-1], energy_map[h-1,w], 
                                energy_map[h-1, w+1]]
            min_energy = np.amin(upper_pixels)
            accumulated_map[h,w] += min_energy
    
    #find the lowest energy path
    min_index = np.argmin(accumulated_map[H-1,:])
    if min_index == 0:
        upper_index = [0, 2]
    elif min_index == W-1:
        upper_index = [W-2, W]
    else:
        upper_index = [min_index-1, min_index+2]
    path = [[H-1, min_index]]
    single_crop[H-1, 0:min_index] = original_img[H-1, 0:min_index]
    single_crop[H-1, min_index:] = original_img[H-1, min_index+1:]
    
    for h in reversed(range(0, H-1)):
        temp_index = np.argmin(accumulated_map[h,upper_index[0]:upper_index[1]])
        min_index = list(range(upper_index[0], upper_index[1]))[temp_index]
        path.append([h, min_index])
        single_crop[h, 0:min_index] = original_img[h, 0:min_index]
        single_crop[h, min_index:] = original_img[h, min_index+1:]
        
        if min_index == 0:
            upper_index = [0, 2]
        elif min_index == W-1:
            upper_index = [W-2, W]
        else:
            upper_index = [min_index -1, min_index +2]
    
    #we can visualize the lowest enery path by plot the white line on the original image
    original_img_with_path = original_img
    if len(original_shape) == 3:
        for p in path:
            original_img_with_path[p[0], p[1]] = [255,0,0]
    else:
        for p in path:
            original_img_with_path[p[0], p[1]] = 255
            
    return single_crop, original_img_with_path, path

def seam_crop(original_img, target_W):
    """
    function of excuting the seam cropping
    """
    _, W = original_img.shape[0], original_img.shape[1]
    out_img = original_img
    crop_time = W - target_W
    for i in range(crop_time):
        out_img, _, _ = path_search_and_crop(out_img)
    return out_img
            
def main(original_img, desired_resolution):
    H,W = original_img.shape[0], original_img.shape[1]
    H_target, W_target = desired_resolution
    
    original_ratio = H/W
    target_ratio = H_target/W_target
    #handle the different desired Height and Width ratio
    if original_ratio > target_ratio:
        seam_H = int(W*target_ratio)
        #handle the RGB image
        if len(original_img.shape) == 3:
            original_img = np.transpose(original_img, [1,0,2])
            seam_cropped_img = np.transpose(seam_crop(original_img, seam_H), [1,0,2])
        else:
            original_img = np.transpose(original_img)
            seam_cropped_img = np.transpose(seam_crop(original_img, seam_H))
        final_img = cv2.resize(seam_cropped_img, (W_target, H_target), interpolation = cv2.INTER_CUBIC)
    elif original_ratio < target_ratio:
        seam_W = int(H/target_ratio)
        seam_cropped_img = seam_crop(original_img, seam_W)
        final_img = cv2.resize(seam_cropped_img, (W_target, H_target), interpolation = cv2.INTER_CUBIC)
    else:
        final_img = cv2.resize(original_img, (W_target, H_target), interpolation = cv2.INTER_CUBIC)
    
    return final_img

if __name__ == '__main__':
    img = cv2.imread('tower.jpg')
    start_time = time.time()
    crop_img = main(img, [487, 487])
    print('Time elapse:', time.time() - start_time)
    cv2.imwrite('cropped_tower.jpg', crop_img)


        