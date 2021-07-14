import numpy as np
import pandas as pd
import skimage.io as io
import PIL as Image
import glob
#import gflags
import sys
import os
import errno
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.path.append(os.path.abspath('Image-segmentation-utilities/'))

from utils import return_padding, pad_image, make_divisor_mask

def sortKeyFunc(s):
    return int(os.path.basename(s).split('_')[-1].split('.')[0])

def stitch_patch(img_name, patch_path, recon_img_path, image_dict, h_stride, w_stride, channel=3, type = 'png'):
    """
        Takes source folder containing patches of Images and reconstruct the original image by recombining
        the patches using naive overlapping assumption without any smoothing and saves them in destination

        NOTE: Patch files should be named like patch_i_j where i = Image number eg. 1,2,3,4 and j = Patch number
              eg. 1,2,3,4 etc. i.e. patch_i_j represent jth patch of ith image
        Params: patch_path -> source folder of patches
                recon_img_path -> destination folder of reconstructed image
                image_dict -> dictionary having image height, image width, patch height, patch width
                            with keys- 'image_height', 'image_width', 'patch_height', 'patch_width'
                h_stride -> 1/overlap taken among adjacent patch along height eg. 0.5 for twice overlap
                w_stride -> 1/overlap taken among adjacent patch along width
                channel  -> number of channel in patches
                type     -> type of patch 'png', 'jpg', 'tif'
    """
    
    if patch_path[-1] != '/':
        patch_path += '/'

    if not os.path.isdir(patch_path):
        raise Exception('patch directory does not exist')
    if not os.path.isdir(recon_img_path):
        print('creating destination folder')
        os.makedirs(destination)

    assert type in ['png', 'jpg', 'tif']
        
    patch_list = []
    #i=1
    #while True:
    patches = sorted(glob.glob(patch_path+'{}_*.png'.format(img_name.split('.')[0])), key=sortKeyFunc)
    
    #if not patches:
    #    break
    patch_list.append(patches)
    for files in patch_list:
        if not files:
            continue
        else:
            patch_height = int(image_dict['patch_height'])
            patch_width  = int(image_dict['patch_height'])
            img_id = img_name#files[0].split('/')[-1].split('_')[1]
            orig_img_height = int(image_dict['image_height'])
            orig_img_width  = int(image_dict['image_width'])
            h_stride = int(h_stride*patch_height)
            w_stride = int(w_stride*patch_width)

            img_dtype = np.uint16
            image     = np.zeros((orig_img_height, orig_img_width, channel), dtype = img_dtype)
            padding   = return_padding(image, patch_height, patch_width)
            image     = pad_image(image, patch_height, patch_width, channel)
            h = 0
            w = 0
            patches = []
            patch_id =0
            for name in files:
                try:
                    if type == 'tif':
                        io.use_plugin('tifffile')
                    patch = io.imread(name)
                    patches.append(patch)
                    if image.dtype != patch.dtype:
                        image = image.astype(patch.dtype, copy=False)                        
                except OSError as e:
                    print(e.errno)
                    print("Some of the patches are corrupted")

            print(len(patches))
            while h <= image.shape[0]-patch_height:
                w = 0
                while w <= image.shape[1]-patch_width:
                    image[h:h+patch_height, w:w+patch_width, :] += patches[patch_id]
                    w = w + w_stride
                    patch_id+=1
                h = h+h_stride
            if(h_stride==w_stride):
                step = patch_height//h_stride
            else:
                print("Unequal strides are not yet suppported")

            mask_height = image.shape[0]//h_stride
            mask_width  = image.shape[1]//w_stride
            divisor_mask = make_divisor_mask(mask_height, mask_width, step)
            #print(divisor_mask)
            print("Divisor mask shape {}".format(divisor_mask.shape))

            h = 0
            w = 0
            mask_h = 0
            mask_w = 0
            print("Image shape {}".format(image.shape))
            while h <= image.shape[0] - h_stride:
                w = 0
                mask_w = 0
                while w <= image.shape[1] - w_stride:
                    image[h:h+h_stride, w:w+w_stride,:] = image[h:h+h_stride, w:w+w_stride,:]#/divisor_mask[mask_h,mask_w]
                    w += w_stride
                    mask_w +=1
                h += h_stride
                mask_h +=1

            #print(image.shape)
            img = image
            #img = image[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1],:]
            print("FinalImage shape{}".format(img.shape))
            assert img.shape == (orig_img_height, orig_img_width, channel)

            if not os.path.isdir(recon_img_path):
                os.mkdir(recon_img_path)

            io.imsave(recon_img_path + '/' + str(img_id) + '.' + type, img)

img_names = os.listdir('Valid_Image')
patch_path = 'Val_Results_exp4/'
res_path = 'Val_Results_reconstructed_exp4'
image_dict = {'image_height':18432, 'image_width':6656, 'patch_height':256, 'patch_width':256}
h_stride = 1.0
w_stride = 1.0
stitch_patch(img_names[0], patch_path, res_path, image_dict, h_stride, w_stride)
image_dict2 = {'image_height':16896, 'image_width':3584, 'patch_height':256, 'patch_width':256}
stitch_patch(img_names[1], patch_path, res_path, image_dict2, h_stride, w_stride)