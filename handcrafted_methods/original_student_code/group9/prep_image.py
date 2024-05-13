### Image preparation module. Used to prepare images for feature extraction.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale

def prep_im_and_mask(im_id, im_dir_path, mask_dir_path, scalar = 1, output_shape = None):
    '''Prepare image and corresponding mask segmentation from test images. 
    Paths to directories containing image and mask files required.
    If parameter scalar is passed, output image will be scaled by it. Defualt 1 retains original size.
    If parameter output_shape is passed_ output image will be resized to it. Defualt None retains original size.

    Args:
        im_id (str): image ID
        im_dir_path (str): image directory path 
        gt_dir_path (str): ground thruth directory path
        scalar (float, optional): rescale coefficient
        output_shape (tuple, optional): resize tuple

    Returns:
        im (numpy.ndarray): image
        mask (numpy.ndarray): mask segmentation.
    '''

    # Read and resize image
    im = plt.imread(im_dir_path + im_id)[:, :, :3] # Some images have fourth, empty color chanel which we slice off here
    im = rescale(im, scalar, anti_aliasing=True, channel_axis = 2) 
    if output_shape != None and scalar == 1:
        im = resize(im, output_shape)

    #Read and resize mask segmentation
    mask = plt.imread(mask_dir_path + im_id[:-4] + "_mask.png")
    if len(mask.shape) == 3:
        mask = mask[:, :, 0] # Some masks have more than 2 dimensions, which we slice off here

    mask = rescale(mask, scalar, anti_aliasing=False)
    
    if output_shape != None and scalar == 1:
        mask = resize(mask, output_shape)

    #Return mask to binary
    binary_mask = np.zeros_like(mask)
    binary_mask[mask > .5] = 1
    mask = binary_mask.astype(int)

    return im, mask

def prep_im(im_id, im_dir_path = "", scalar = 1, output_shape = None):
    '''Prepare image from im_id and optional dictory path.
    If directory path is not passed, the whole filepath, including filetype notation, 
    should be given as im_id. 
    If parameter scalar is passed, output image will be scaled by it. Defualt 1 retains original size.
    If parameter output_shape is passed_ output image will be resized to it. Defualt None retains original size.
    
    Args:
        im_id (str): image ID
        im_dir_path (str, optional): image directory path
        scalar (float, optional): rescale coefficient
        output_shape (tuple, optional): resize tuple

    Returns:
        im (numpy.ndarray): image.
    '''

    # Read and resize image
    if im_dir_path == "":
        im = plt.imread(im_id)[:, :, :3] #Some images have fourth, empty color chanel which we slice of here
    else:
        im = plt.imread(im_dir_path + im_id)[:, :, :3] #Some images have fourth, empty color chanel which we slice of here
    im = rescale(im, scalar, anti_aliasing=True, channel_axis = 2) #IDWE: Use channel_axis=2 to prevent picture from being turned bianry when rescaled
    if output_shape != None and scalar == 1:
        im = resize(im, output_shape)

    return im

def prep_mask(im_id, mask_dir_path = "", scalar = 1, output_shape = None):
    '''Prepare mask segmentaion from im_id and optional dictory path.
    If directory path is not passed, the whole filepath, including filetype notation, 
    should be given as im_id. 
    If parameter scalar is passed, output image will be scaled by it. Defualt 1 retains original size.
    If parameter output_shape is passed_ output image will be resized to it. Defualt None retains original size.
    
    Args:
        im_id (str): image ID
        mask_dir_path (str, optional): mask directory path
        scalar (float, optional): rescale coefficient
        output_shape (tuple, optional): resize tuple

    Returns:
        mask (numpy.ndarray): mask segmentation.
    '''

    # Read and resize image
    if mask_dir_path == "":
        mask = plt.imread(im_id)
    else:
        mask = plt.imread(mask_dir_path + im_id[:-4] + ".png")

    if len(mask.shape) == 3:
        mask = mask[:, :, 0] # Some masks have more than 2 dimensions, which we slice off here
    
    mask = rescale(mask, scalar, anti_aliasing=False)
    if output_shape != None and scalar == 1:
        mask = resize(mask, output_shape)

    # Return mask to binary
    binary_mask = np.zeros_like(mask)
    binary_mask[mask > .5] = 1
    mask = binary_mask.astype(int)

    return mask