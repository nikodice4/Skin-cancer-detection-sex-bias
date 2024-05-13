import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# necessairy for windows
os.environ['OMP_NUM_THREADS'] = '1'

from skimage.transform import resize, rescale
from skimage import morphology
from skimage import color


from glob import glob
from tqdm import tqdm

import cv2

import sys

from handcrafted_methods import extract_features as feature

import csv
import shutil
from PIL import Image

############################################# Group 5 #############################################

def superpose_segmentation(img, mask):

    # Get a list of all the images to be processed
    images = os.listdir(img)

    df = pd.DataFrame()

    filename = []
    pigment_network_coverage = []
    blue_veil_pixels = []
    globules_count = []
    streaks_irregularity = []
    irregular_pigmentation_coverage = []
    regression_pixels = []

    for file in images:
        # Check if the file has a corresponding segmentation image
        segmentation_file = os.path.join(mask, file.replace('.png', '_mask.png'))

        # Full image path
        image_path = os.path.join(img, file)
 
        if os.path.exists(segmentation_file):
            try:
                # Open the normal image and the segmentation image
                normal_image = Image.open(os.path.join(img, file)).convert("RGBA")
                segmentation_image = Image.open(segmentation_file).convert("RGBA")

                # Resize the images to 256x256 pixels
                normal_image = normal_image.resize((256, 256))
                segmentation_image = segmentation_image.resize((256, 256))

                # Invert the segmentation image
                inverted_segmentation = Image.eval(segmentation_image, lambda x: 255 - x)

                # Create a binary mask from the inverted segmentation image
                new_mask = inverted_segmentation.split()[0].point(lambda x: 255 if x == 0 else 0).convert("L")

                # Apply the mask to the normal image
                normal_image.putalpha(new_mask)

                # Create a black background
                background = Image.new("RGBA", normal_image.size, (0, 0, 0, 255))

                # Composite the normal image with the black background
                result = Image.alpha_composite(background, normal_image)

                features = extracting_features(image_path, result)
               
                filename.append(features["filename"])
                pigment_network_coverage.append(features["pigment_network_coverage"])
                blue_veil_pixels.append(features["blue_veil_pixels"])
                globules_count.append(features["globules_count"])
                streaks_irregularity.append(features["streaks_irregularity"])
                irregular_pigmentation_coverage.append(features["irregular_pigmentation_coverage"])
                regression_pixels.append(features["regression_pixels"])

                print(f"Superposed {file}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        else:
            print(f"No segmentation image found for {file}")

    df["filename"] = filename
    df["pigment_network_coverage"] = pigment_network_coverage
    df["blue_veil_pixels"] = blue_veil_pixels
    df["globules_count"] = globules_count
    df["streaks_irregularity"] = streaks_irregularity
    df["irregular_pigmentation_coverage"] = irregular_pigmentation_coverage
    df["regression_pixels"] = regression_pixels

    return df

def extracting_features(image_path, image):
    """
    Extract features from an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: Dictionary containing the extracted features.
    """
    
    # Convert PIL image to numpy array
    image = np.array(image)

    # Convert from RGBA to RGB, since that is what the student code is written to work with
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    features = {}
    features['filename'] = os.path.basename(image_path)
    features['pigment_network_coverage'] = feature.measure_pigment_network(image)
    features['blue_veil_pixels'] = feature.measure_blue_veil(image)
    features['vascular_pixels'] = feature.measure_vascular(image)
    features['globules_count'] = feature.measure_globules(image)
    features['streaks_irregularity'] = feature.measure_streaks(image)
    features['irregular_pigmentation_coverage'] = feature.measure_irregular_pigmentation(image)
    features['regression_pixels'] = feature.measure_regression(image)

    return features

def extractFeaturesGroup5(imgPath, maskPath):

    # Provide the paths to the folders containing the images
    images = imgPath
    segmentation_folder = maskPath

    # Pre-process the images and extract features
    dataframe = superpose_segmentation(images, segmentation_folder)

    return dataframe
    
############################################# Group 4 #############################################


def extractFeaturesGroup4(mask_path, img_path, metadata):
    print("extractFeatures called")  # Debugging print
    
    # Read metadata
    metadata_df = metadata
    
    names = [s.split(os.sep)[-1] for s in sorted(glob(f"{mask_path}/*_mask.png"))]
    
    with open('debug.txt', 'w') as file:
        file.write(str(os.path.exists(mask_path))) # Should return True
        file.write(str(os.path.exists(img_path)))

    df = pd.DataFrame()

    image_names = []
    compactness = []
    avg_red_channel = []
    avg_green_channel = []
    avg_blue_channel = []
    multicolor_rate = []
    asymmetry = []

    average_hue = []
    average_saturation = []
    average_value = []
    
    for name in tqdm(names):
        # Remove '_segmentation.png' from the mask filename to match the original image filename
        base_name = name.replace('_mask.png', '.png')
        save_name = base_name.replace('.png', '')

        with open('debug.txt', 'w') as file:
            file.write(save_name)
        
        # Skip if base_name not in metadata
        if not metadata_df[metadata_df['img_id'].str.contains(save_name)].empty:
            #print("skips because meta does not contrain that image")
            
            mask_file_path = os.path.join(mask_path, name)
            image_file_path = os.path.join(img_path, base_name)

            try:
                mask = plt.imread(mask_file_path)
                image = plt.imread(image_file_path)[:, :, :3]
            except FileNotFoundError as e:
                with open('debug.txt', 'w') as file:
                    file.write(f"File not found: {e}")
                continue
            
            # Ensure mask is binary
            mask = binary_mask(mask, image.shape)
            if np.sum(mask) == 0:
                with open('debug.txt', 'w') as file:
                    file.write(f"Empty mask for image {name}")
                continue

            image_names.append(save_name)

            r, g, b = averageColor(image, mask)
            avg_red_channel.append(r)
            avg_green_channel.append(g)
            avg_blue_channel.append(b)
            compactness.append(feature.get_compactness(mask))
            multicolor_rate.append(feature.get_multicolor_rate(image, mask, 3))
            asymmetry.append(feature.get_asymmetry(mask))
            
            h, s, v = averageColorHSV(image, mask)
            average_hue.append(h)
            average_saturation.append(s)
            average_value.append(v)
        else:
            with open('debug.txt', 'w') as file:
                file.write(f"Skipping {base_name} as it's not found in metadata.")

    df["image_names"] = image_names
    df["compactness"] = compactness
    df["avg_red_channel"] = avg_red_channel
    df["avg_green_channel"] = avg_green_channel
    df["avg_blue_channel"] = avg_blue_channel
    df["multicolor_rate"] = multicolor_rate
    df["asymmetry"] = asymmetry
    df["average_hue"] = average_hue
    df["average_saturation"] = average_saturation
    df["average_value"] = average_value

    return df

def binary_mask(mask, shape, threshold = 0.5):
        mask = resize(mask, output_shape= shape)
        mask = color.rgb2gray(mask)
        mask[mask < threshold] = 0
        mask[mask > threshold] = 1
        return mask


def averageColor(img, mask):
    # Make a writable copy of the img array
    img = np.copy(img)
    
    img[mask == 0] = 0
    tot_pixels = np.sum(mask)
    red_avg = np.sum(img[:, :, 0]) / tot_pixels
    green_avg = np.sum(img[:, :, 1]) / tot_pixels
    blue_avg = np.sum(img[:, :, 2]) / tot_pixels
    pixel_color = np.array([red_avg, green_avg, blue_avg])
    return pixel_color

def averageColorHSV(img, mask):
    img = color.rgb2hsv(img)

    img[mask == 0] = 0
    tot_pixels = np.sum(mask)
    red_avg = np.sum(img[:, :, 0]) / tot_pixels
    green_avg = np.sum(img[:, :, 1]) / tot_pixels
    blue_avg = np.sum(img[:, :, 2]) / tot_pixels
    pixel_color = np.array([red_avg, green_avg, blue_avg])
    return pixel_color



############################################# Group 9 #############################################

def ProcessImages(file_data, image_folder, mask_folder, feature_names):
    '''
    Process images and extract features.
    
    Args:
        file_data (str): File path or file object containing metadata.
        image_folder (str): Path to the folder containing the images.
        mask_folder (str): Path to the folder containing the masks.
        feature_names (list): List of feature names.
    '''
    df = file_data
    print("2")
    print("3")
    # Features to extract
    features_n = len(feature_names)
    
    # Initialize an empty array for features; this may change size dynamically
    features = []
    img_ids = []
    # Extract features
    for i, id in enumerate(df['img_id']):
        id_with_extension = id.replace('.png', '_mask.png') # Assuming .png is the correct extension print(id_with_extension)
        image_path = os.path.join(image_folder, id)
        mask_path = os.path.join(mask_folder, id_with_extension)

        if not os.path.exists(image_path):
            print(f"Image for {id} not found, skipping...")

            continue

        if not os.path.exists(mask_path):
            print(f"Mask for {id} not found, skipping...")

            continue

        try:
            im, mask = prep_im_and_mask(id, image_folder, mask_folder)
            x = extract_features_group9(im, mask)

            img_ids.append(id)
            features.append(x)  # Append the features for each image

            print(f"Processed {i+1} out of {len(df)} images")
        except Exception as e:
            print(f"Error processing {id_with_extension}: {e}, skipping...")
            continue

    # Check if we have processed any images
    if img_ids:
        # Convert the list of features to a numpy array
        features_array = np.array(features, dtype=np.float32)
        
        # Insert image ids
        df_features = pd.DataFrame(features_array, columns=feature_names)
        df_features.insert(0, 'ID', img_ids)

        return df_features
    else:
        print("No images were processed.")

def prep_im_and_mask(im_id, im_dir_path, mask_dir_path, scalar = 1, output_shape = None):
    '''Prepare image and corresponding mask segmentation from test images. 
    Paths to directories containing image and mask files required.
    If parameter scalar is passed, output image will be scaled by it. Defualt 1 retains original size.
    If parameter output_shape is passed_ output image will be resized to it. Default None retains original size.

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

def extract_features_group9(im, im_mask):
    '''
    Extract a set of features from an image and its corresponding mask.

    Args:
        im (array-like): Input image.
        im_mask (array-like): Mask corresponding to the image.

    Returns:
        list: List of extracted features, including asymmetry measures, color variances, color dominance, compactness,
              convexity, and relative color scores.

    '''
    # Assymmetry
    mean_asym = feature.mean_asymmetry(im_mask, 4)
    best_asym = feature.best_asymmetry(im_mask, 4)
    worst_asym = feature.worst_asymmetry(im_mask, 4)

    # Color variance
    segments = feature.slic_segmentation(im, im_mask, n_segments=250)
    red_var, green_var, blue_var = feature.rgb_var(im, segments)
    hue_var, sat_var, val_var = feature.hsv_var(im, segments)

    # Color dominance
    dom_colors = feature.color_dominance(im, im_mask, clusters=5, include_ratios=True) # Extract five most dominent colors, sorted by percentage of total area
    dom_hue, dom_sat, dom_val = dom_colors[0][1]     

    # Compactness
    compactness = feature.compactness_score(im_mask)

    # Convexity
    convexity = feature.convexity_score(im_mask)

    # Relative color scores
    F1, F2, F3, F10, F11, F12 = feature.get_relative_rgb_means(im, segments)

    return [mean_asym, best_asym, worst_asym, red_var, green_var, \
        blue_var, hue_var, sat_var, val_var, dom_hue, dom_sat, \
        dom_val, compactness, convexity, F1, F2, F3, F10, F11, F12]


def extractFeaturesGroup9(df, img_path, mask_path):
    file_data = df
    image_folder = img_path
    mask_folder = mask_path

    feature_names = ['mean_asymmetry', 'best_asymmetry', 'worst_asymmetry', 'red_var', 'green_var', \
     'blue_var', 'hue_var', 'sat_var', 'val_var', 'dom_hue', 'dom_sat', 'dom_val', \
     'compactness', 'convexity', 'F1', 'F2', 'F3', 'F10', 'F11', 'F12']
    
    print("1")
    df_features = ProcessImages(file_data, image_folder, mask_folder, feature_names)
    return df_features

############################################# Extraction #############################################

def getFeaturesGroup5(img, mask): # 7-point checklist 
    
    feature_df = extractFeaturesGroup5(
        img, mask
    )

    feature_df = feature_df.rename(columns={'filename': 'img_id'})

    return feature_df

def getFeaturesGroup4(df, img, mask):

    feature_df = extractFeaturesGroup4(
        mask, img, df
    )

    feature_df = feature_df.rename(columns={'image_names': 'img_id'})
    feature_df['img_id'] = feature_df['img_id'] + '.png' # these definetely need .png

    return feature_df

def getFeaturesGroup9(df, img, mask):
    
    feature_df = extractFeaturesGroup9(
        df, img, mask
    )
    
    feature_df = feature_df.rename(columns={'ID': 'img_id'})

    return feature_df


def getAllFeatures(df, img, mask):
    df1 = getFeaturesGroup5(img, mask)
    df2 = getFeaturesGroup4(df, img, mask)
    df3 = getFeaturesGroup9(df, img, mask)

    df = pd.merge(df, df1, on='img_id', how='inner')
    df = pd.merge(df, df2, on='img_id', how='inner')
    df = pd.merge(df, df3, on='img_id', how='inner')

    return df
