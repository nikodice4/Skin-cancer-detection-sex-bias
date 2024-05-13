import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from skimage.transform import resize
from skimage.transform import rotate
from skimage import morphology
from skimage import color

from glob import glob
from tqdm import tqdm

import cv2
from sklearn.cluster import KMeans

import os

def features2Dataframe(image, mask):
    df = pd.DataFrame()
    print("hi2")
    image = image[:, :, :3]
    mask = binary_mask(mask, image.shape)

    r, g, b = averageColor(image, mask)

    df["compactness"] = [get_compactness(mask)]
    df["multicolor_rate"] = [get_multicolor_rate(image, mask, 3)]
    df["asymmetry"] = [get_asymmetry(mask)]
    df["avg_red_channel"] = [r]
    df["avg_green_channel"] = [g]
    df["avg_blue_channel"] = [b]
    return df


def extractFeatures(mask_path, img_path, metadata):
    print("extractFeatures called")  # Debugging print
    
    # Read metadata
    metadata_df = pd.read_csv(metadata)
    
    names = [s.split(os.sep)[-1] for s in sorted(glob(f"{mask_path}/*_mask.png"))]
    
    print(os.path.exists(mask_path))  # Should return True
    print(os.path.exists(img_path))

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
        print(save_name)
        
        # Skip if base_name not in metadata
        if not metadata_df[metadata_df['img_id'].str.contains(save_name)].empty:
            #print("skips because meta does not contrain that image")
            
            mask_file_path = os.path.join(mask_path, name)
            image_file_path = os.path.join(img_path, base_name)

            try:
                mask = plt.imread(mask_file_path)
                image = plt.imread(image_file_path)[:, :, :3]
            except FileNotFoundError as e:
                print(f"File not found: {e}")
                continue
            
            # Ensure mask is binary
            mask = binary_mask(mask, image.shape)
            if np.sum(mask) == 0:
                print(f"Empty mask for image {name}")
                continue

            image_names.append(save_name)

            r, g, b = averageColor(image, mask)
            avg_red_channel.append(r)
            avg_green_channel.append(g)
            avg_blue_channel.append(b)
            compactness.append(get_compactness(mask))
            multicolor_rate.append(get_multicolor_rate(image, mask, 3))
            asymmetry.append(get_asymmetry(mask))
            
            h, s, v = averageColorHSV(image, mask)
            average_hue.append(h)
            average_saturation.append(s)
            average_value.append(v)
        else:
            print(f"Skipping {base_name} as it's not found in metadata.")

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

def readMetadata(self, path):
    return pd.read_csv(path)

def get_compactness(mask):
    # mask = color.rgb2gray(mask)
    area = np.sum(mask)

    struct_el = morphology.disk(3)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.sum(mask - mask_eroded)

    return perimeter**2 / (4 * np.pi * area)

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

def get_com_col(cluster, centroids):
    com_col_list = []
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)], key= lambda x:x[0])
    start = 0
    for percent, color in colors:
        if percent > 0.08:
            com_col_list.append(color)
        end = start + (percent * 300)
        cv2.rectangle(
            rect,
            (int(start), 0),
            (int(end), 50),
            color.astype("uint8").tolist(),
            -1,
        )
        start = end
    return com_col_list

def get_multicolor_rate(im, mask, n):
    # mask = color.rgb2gray(mask)
    im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
    mask = resize(
        mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=True
    )
    im2 = im.copy()
    im2[mask == 0] = 0

    columns = im.shape[0]
    rows = im.shape[1]
    col_list = []
    for i in range(columns):
        for j in range(rows):
            if mask[i][j] != 0:
                col_list.append(im2[i][j] * 256)

    if len(col_list) == 0:
        return ""

    cluster = KMeans(n_clusters=n, n_init=10).fit(col_list)
    com_col_list = get_com_col(cluster, cluster.cluster_centers_)

    dist_list = []
    m = len(com_col_list)

    if m <= 1:
        return ""

    for i in range(0, m - 1):
        j = i + 1
        col_1 = com_col_list[i]
        col_2 = com_col_list[j]
        dist_list.append(
            np.sqrt(
                (col_1[0] - col_2[0]) ** 2
                + (col_1[1] - col_2[1]) ** 2
                + (col_1[2] - col_2[2]) ** 2
            )
        )
    return np.max(dist_list)

def midpoint(mask):
        summed = np.sum(mask, axis=0)
        half_sum = np.sum(summed) / 2
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                return i
def crop(mask):
        mid = midpoint(mask)
        y_nonzero, x_nonzero = np.nonzero(mask)
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
        x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
        x_dist = max(np.abs(x_lims - mid))
        x_lims = [mid - x_dist, mid+x_dist]
        return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]

def get_asymmetry(mask):
    # mask = color.rgb2gray(mask)
    scores = []
    for _ in range(6):
        segment = crop(mask)
        (np.sum(segment))
        scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / (np.sum(segment)))
        mask = rotate(mask, 30)
    return sum(scores) / len(scores)

def main():
    df = extractFeatures(mask_path="masks/allMasks/", img_path="pad-ufes/images/", metadata="pad-ufes/metadata.csv")
    df.to_csv("student_code/group4", index=False)

if __name__ == "__main__":
    main()
