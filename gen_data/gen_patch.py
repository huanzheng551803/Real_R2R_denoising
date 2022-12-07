import glob
import os
import cv2
import numpy as np

images_dir = './srgb/*_GT_RAW_*_srgb_r2r_input.png'
image_paths = sorted(glob.glob(images_dir))
train_num = 0

os.makedirs('../sidd_dataset/train_patches/',exist_ok=True)
for k in range(len(image_paths)):
    image_path_ = image_paths[k]
    r2r_input = cv2.imread(image_path_,-1)
    r2r_output = cv2.imread(image_path_.replace('_srgb_r2r_input','_srgb_r2r_output'),-1)

    H,W = r2r_input.shape[0],r2r_input.shape[1]
    patch_size = 128
    stride=128

    len1 =  int(np.ceil((H-patch_size)/stride)+1)
    len2 = int(np.ceil((W-patch_size)/stride)+1)
    
    for i in range(len1):
        for j in range(len2):
            if j == len2-1:
                r2r_input_patch = r2r_input[:,-patch_size:,:]
                r2r_output_patch = r2r_output[:,-patch_size:,:]
            else:
                r2r_input_patch = r2r_input[:,stride*j:stride*j+patch_size:,:]
                r2r_output_patch = r2r_output[:,stride*j:stride*j+patch_size:,:]
            if i == len1-1:
                r2r_input_patch = r2r_input_patch[-patch_size:,:,:]
                r2r_output_patch = r2r_output_patch[-patch_size:,:,:]
            else:
                r2r_input_patch = r2r_input_patch[stride*i:stride*i+patch_size,:,:]
                r2r_output_patch = r2r_output_patch[stride*i:stride*i+patch_size,:,:]

            cv2.imwrite('../sidd_dataset/train_patches/%d_r2r_input.png'%train_num,r2r_input_patch)
            cv2.imwrite('../sidd_dataset/train_patches/%d_r2r_output.png'%train_num,r2r_output_patch)
            
            train_num+=1
    print(image_path_,len1,len2,r2r_input.shape,train_num)
