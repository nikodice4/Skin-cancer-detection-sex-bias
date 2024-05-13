# We use PyTorch instead of TorchIO because we are dealing with smartphone images,
# even though they intuitively seem like medical images.
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import os
import pandas as pd

# Need to set this environment variable to avoid a warning from OpenMP to run main (not needed for normal use)
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

src_path = '../data/images/lesion_images/' # need to navigate out one folder because the script is run from a notebook, which is located on step in from the root folder
dst_path_once = '../data/images/augmented_images_once/'
dst_path_50_50 = '../data/images/augmented_images_50_50/'
mask_dir = '../data/images/lesion_masks/'
dst_mask_path_once = '../data/images/augmented_masks_once/'
dst_mask_path_50_50 = '../data/images/augmented_masks_50_50/'

rng = np.random.default_rng(1173).bit_generator

def augment_image (img_path):
    img = Image.open(img_path)
    mask = Image.open(generate_mask_path(img_path))

    transforms = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.GaussianBlur(15, sigma=(0.1, 8.0)),
        v2.RandomAdjustSharpness(sharpness_factor=5)
    ])
    img, mask = transforms(img, mask)
    return img, mask


def augment_images_once (img_list): # takes list of img_ids and saves augmented imgs
    for index, element in enumerate(img_list):
        filepath = src_path + element
        filename = 'aug_' + element

        augmented_img, augmented_mask = augment_image(filepath)
        
        augmented_img.save(os.path.join(dst_path_once, filename))
        augmented_mask.save(os.path.join(dst_mask_path_once, filename[:-4] + '_mask.png'))

def augment_images_50_50 (img_list): # takes list of img_ids, saves augmented imgs and adds iteration of augmented image
    for index, element in enumerate(img_list):
        filepath = src_path + element[:-6] + element[-4:] # remove the suffix denoting the iteration of the image but still keeping .png
        filename = 'aug_' + element

        augmented_img, augmented_mask = augment_image(filepath)
        
        augmented_img.save(os.path.join(dst_path_50_50, filename))
        
        augmented_img.save(os.path.join(dst_path_50_50, filename))
        augmented_mask.save(os.path.join(dst_mask_path_50_50, filename[:-4] + '_mask.png'))

def modify_dataframe_once (df, img_list): # takes a dataframe and adds metadata for augmented images
    for index, element in enumerate(img_list):
        
        img_id = element[:-6] + element[-4:] # remove the suffix denoting the iteration of the image but still keeping .png
        suffix = '000' + element[-5] # get the suffix denoting the iteration of the image and add it to 000 so it is obvious which lesion ids are augmented and prevent duplicates


        matched_row = df[df['img_id'] == element].copy()
        matched_row['img_id'] = 'aug_' + element
        matched_row['lesion_id'] = (matched_row['lesion_id'].astype(str) + suffix).astype(int) # convert to str to add suffix and then back to int

        df = pd.concat([df, matched_row], ignore_index=True)

    return df    

def modify_dataframe_50_50 (df, img_list): # takes a dataframe and adds metadata for augmented images but adds iteration of augmented image
    for index, element in enumerate(img_list):
        
        img_id = element[:-6] + element[-4:] # remove the suffix denoting the iteration of the image but still keeping .png
        suffix = '000' + element[-5] # get the suffix denoting the iteration of the image and add it to 000 so it is obvious which lesion ids are augmented

        matched_row = df[df['img_id'] == img_id].copy()
        matched_row['img_id'] = 'aug_' + element
        matched_row['lesion_id'] = (matched_row['lesion_id'].astype(str) + suffix).astype(int) # convert to str to add suffix and then back to int

        df = pd.concat([df, matched_row], ignore_index=True)

    return df

def sample50_50(df, amount_sample, length_class):
    df_50_50 = pd.DataFrame()
    loop_length = (amount_sample // length_class) + 1
    for i in range(loop_length, 0, -1):
        if i > 1:
            sampled_df = df.sample(n=length_class, random_state=rng).copy()  # Make a copy of sampled DataFrame
            sampled_df['img_id'] = sampled_df['img_id'].apply(lambda x: f"{x[:-4]}_{i}{x[-4:]}")
            df_50_50 = pd.concat([df_50_50, sampled_df], sort=False) 
        if i == 1:
            sampled_df = df.sample(n=amount_sample-length_class, random_state=rng).copy()  # Make a copy of sampled DataFrame
            sampled_df['img_id'] = sampled_df['img_id'].apply(lambda x: f"{x[:-4]}_{i}{x[-4:]}")
            df_50_50 = pd.concat([df_50_50, sampled_df], sort=False)  
    return df_50_50        



# Takes a dataframe and a binary feature, calculates the number of images needed to balance the dataset, and augments the images to 
# balance the dataset, then returns the dataframe with metadata for augmented images gathered from their original counterparts
def balance_dataset(df, feature):

    counts = df[feature].value_counts()

    if ( counts[0] > counts[1] ):
        num_augment = counts[0] - counts[1]
        df_augmented = df[df[feature] == 1]
        
        sampled_data_once = df_augmented.sample(n=counts[1], random_state=rng).copy() # sample for all avaiable images (less than 50/50)
        image_ids_once = sampled_data_once['img_id'].tolist()

        sampled_data_50_50 = sample50_50(df_augmented, counts[0], counts[1]) # sample for images the minimum needed to balance the dataset to 50/50
        image_ids_50_50 = sampled_data_50_50['img_id'].tolist()

        # augment_images_once(image_ids_once)
        # augment_images_50_50(image_ids_50_50)
        final_df_once_augmented = modify_dataframe_once(df, image_ids_once)
        final_df_50_50 = modify_dataframe_50_50(df, image_ids_50_50)



    elif ( counts[1] > counts[0] ):
        num_augment = counts[1] - counts[0]
        df_augmented = df[df[feature] == 0]

        sampled_data_once = df_augmented.sample(n=counts[0], random_state=rng).copy() # sample for all avaiable images (less than 50/50)
        image_ids_once = sampled_data_once['img_id'].tolist()

        sampled_data_50_50 = sample50_50(df_augmented, num_augment, counts[0]) # sample for images the minimum needed to balance the dataset to 50/50
        image_ids_50_50 = sampled_data_50_50['img_id'].tolist()

        augment_images_once(image_ids_once)
        augment_images_50_50(image_ids_50_50)
        final_df_once_augmented = modify_dataframe_once(df, image_ids_once)
        final_df_50_50 = modify_dataframe_50_50(df, image_ids_50_50)

    return final_df_once_augmented, final_df_50_50


def generate_mask_path(img_path):
    directory, filename = os.path.split(img_path)
    
    filename_no_ext, ext = os.path.splitext(filename) # Extract the filename without extension
    
    mask_filename = filename_no_ext + "_mask.png"  # Construct the mask filename with _mask suffix and .png extension
    
    mask_path = os.path.join(mask_dir, mask_filename)  # Join the directory and filename to get the full mask path
    
    return mask_path


# Show the original image and augmented image side by side, to test a sample of the augmentation
if __name__ == '__main__':
    img_path = 'test/test_images/PAT_34_47_108.png'
    img = Image.open(img_path)

    augmented_img = augment_image(img_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(augmented_img)
    axes[1].set_title('Augmented Image')
    axes[1].axis('off')
    plt.show()