import cv2
import glob
import os
import utils
import numpy as np
'''
mask_path = '../Train_Mask'
img_list = os.listdir(mask_path)
dest_path = '../Dataset4/Train_Mask'
img_name = []

for img in img_list:
    image = cv2.imread(os.path.join(mask_path, img))
    #print(image)
    patches, params = utils.create_patches(image, 512, 512, 0.5, 0.5)
    i = 0
    for patch in patches:
        i+=1
        if len(np.unique(patch)) == 1:
            num = np.random.rand()
            if num>0.8:
                cv2.imwrite(os.path.join(dest_path, img.split('.')[0]+'_'+str(i)+'.png'), patch)
                img_name.append(img.split('.')[0].split('_')[0]+'_'+str(i)+'.png')
        else:
            cv2.imwrite(os.path.join(dest_path, img.split('.')[0]+'_'+str(i)+'.png'), patch)
            img_name.append(img.split('.')[0].split('_')[0]+'_'+str(i)+'.png')
            

img_path = '../Train_Image'
img_list = os.listdir(img_path)
dest_path = '../Dataset4/Train_Image'


for img in img_list:
    image = cv2.imread(os.path.join(img_path, img))
    #print(image)
    patches, params = utils.create_patches(image, 512, 512, 0.5, 0.5)
    i = 0
    for patch in patches:
        i+=1
        if len(np.unique(patch)) == 1:
            #if np.unique(patch)[0] == 255:
            if any(img.split('.')[0]+'_'+str(i)+'.png' in x for x in img_name):
                    cv2.imwrite(os.path.join(dest_path, img.split('.')[0]+'_'+str(i)+'.png'), patch)
                    img_name.append(img.split('.')[0]+'_'+str(i)+'.png')
        else:
            if any(img.split('.')[0]+'_'+str(i)+'.png' in x for x in img_name):
                cv2.imwrite(os.path.join(dest_path, img.split('.')[0]+'_'+str(i)+'.png'), patch)
'''                
mask_path = '../Valid_Mask'
img_list = os.listdir(mask_path)
dest_path = '../Dataset4/Val_Mask_no_overlap'
img_name = []

for img in img_list:
    image = cv2.imread(os.path.join(mask_path, img))
    #print(image)
    patches, params = utils.create_patches(image, 512, 512, 1.0, 1.0)
    i = 0
    for patch in patches:
        i+=1
        cv2.imwrite(os.path.join(dest_path, img.split('.')[0].split('_')[0]+'_'+str(i)+'.png'), patch)
        #img_name.append(img.split('.')[0].split('_')[0]+'_'+str(i)+'.png')

img_path = '../Valid_Image'
img_list = os.listdir(img_path)
dest_path = '../Dataset4/Val_Image_no_overlap'


for img in img_list:
    image = cv2.imread(os.path.join(img_path, img))
    #print(image)
    patches, params = utils.create_patches(image, 512, 512, 1.0, 1.0)
    i = 0
    for patch in patches:
        i+=1
        cv2.imwrite(os.path.join(dest_path, img.split('.')[0]+'_'+str(i)+'.png'), patch)