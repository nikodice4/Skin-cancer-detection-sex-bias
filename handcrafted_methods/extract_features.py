import cv2
import numpy as np
from math import sqrt, floor, ceil, nan, pi
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.transform import rotate
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar, circstd
from statistics import variance, stdev
from scipy.spatial import ConvexHull



############################################## Group 5 ############################################################################################
def measure_pigment_network(image):
    """
    Measure the coverage percentage of the pigment network in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        float: Coverage percentage of the pigment network.
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab_image)

    enhanced_l_channel = cv2.equalizeHist(l_channel)
    _, binary_mask = cv2.threshold(enhanced_l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    total_pixels = np.prod(binary_mask.shape[:2])
    pigment_pixels = np.count_nonzero(binary_mask)
    coverage_percentage = (pigment_pixels / total_pixels) * 100

    return coverage_percentage


def measure_blue_veil(image):
    """
    Measure the number of pixels exhibiting blue veil in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        int: Number of pixels with blue veil.
    """
    height, width, _ = image.shape
    count = 0

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]

            if b > 60 and (r - 46 < g) and (g < r + 15):
                count += 1

    return count


def measure_vascular(image):
    """
    Measure the presence of vascular structures in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        int: Number of pixels representing vascular structures.
    """
    red_channel = image[:, :, 0]
    enhanced_red_channel = exposure.adjust_gamma(red_channel, gamma=1)
    enhanced_image = image.copy()
    enhanced_image[:, :, 0] = enhanced_red_channel
    hsv_img = color.rgb2hsv(enhanced_image)

    lower_red1 = np.array([0, 40/100, 00/100])
    upper_red1 = np.array([25/360, 1, 1])
    mask1 = np.logical_and(np.all(hsv_img >= lower_red1, axis=-1), np.all(hsv_img <= upper_red1, axis=-1))

    lower_red2 = np.array([330/360, 40/100, 00/100])  # Lower limit for red hue, saturation, and value
    upper_red2 = np.array([1, 1, 1])  # Upper limit for red hue, saturation, and value
    mask2 = np.logical_and(np.all(hsv_img >= lower_red2, axis=-1), np.all(hsv_img <= upper_red2, axis=-1))

    mask = np.logical_or(mask1, mask2)

    return np.sum(mask)


def measure_globules(image):
    """
    Measure the number of globules in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        int: Number of globules.
    """
    image_gray = rgb2gray(image)
    inverted_image = 1 - image_gray

    blobs_doh = blob_log(inverted_image, min_sigma=1, max_sigma=4, num_sigma=50, threshold=.05)
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)
    blob_amount = len(blobs_doh)

    return blob_amount


def measure_streaks(image):
    """
    Measure the irregularity of streaks in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        float: Irregularity measure of streaks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lesion_area = cv2.contourArea(contours[0])
    border_perimeter = cv2.arcLength(contours[0], True)
    if lesion_area == 0:
        irregularity = 0
    else:
        irregularity = (border_perimeter ** 2) / (4 * np.pi * lesion_area)

    return irregularity


def measure_irregular_pigmentation(image):
    """
    Measure the coverage percentage of irregular pigmentation in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        float: Coverage percentage of irregular pigmentation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(gray)
    binary = gray > threshold
    labeled_image = label(binary)

    min_rows, min_cols, max_rows, max_cols = [], [], [], []

    for region in regionprops(labeled_image):
        area = region.area
        perimeter = region.perimeter

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))

        if circularity < 0.6:
            min_row, min_col, max_row, max_col = region.bbox
            min_rows.append(min_row)
            min_cols.append(min_col)
            max_rows.append(max_row)
            max_cols.append(max_col)

    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total_pixels = np.prod(binary_mask.shape[:2])
    irregular_pixels = np.count_nonzero(binary_mask)
    coverage_percentage = (irregular_pixels / total_pixels) * 100

    return coverage_percentage


def measure_regression(image):
    """
    Measure the number of pixels representing regression structures in an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        int: Number of pixels representing regression structures.
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 150])
    upper_color = np.array([180, 30, 255])
    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    num_pixels = cv2.countNonZero(mask)

    return num_pixels



############################################## Group 4 ##########################################################################

def get_compactness(mask):
    # mask = color.rgb2gray(mask)
    area = np.sum(mask)

    struct_el = morphology.disk(3)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.sum(mask - mask_eroded)

    return perimeter**2 / (4 * np.pi * area)

def get_asymmetry(mask):
    # mask = color.rgb2gray(mask)
    scores = []
    for _ in range(6):
        segment = crop(mask)
        (np.sum(segment))
        scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / (np.sum(segment)))
        mask = rotate(mask, 30)
    return sum(scores) / len(scores)

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

def crop(mask):
        mid = midpointGroup4(mask)
        y_nonzero, x_nonzero = np.nonzero(mask)
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
        x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
        x_dist = max(np.abs(x_lims - mid))
        x_lims = [mid - x_dist, mid+x_dist]
        return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]

def midpointGroup4(mask):
        summed = np.sum(mask, axis=0)
        half_sum = np.sum(summed) / 2
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                return i
            
###################################### Group 9 ####################################################################

def cut_mask(mask):
    '''Cut empty space from mask array such that it has smallest possible dimensions.

    Args:
        mask (numpy.ndarray): mask to cut

    Returns:
        cut_mask_ (numpy.ndarray): cut mask    
    '''
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_

def cut_im_by_mask(image, mask):
    '''Cut image array such that it has smallest possible dimensions in accordance with its mask.

    Args:
        image (numpy.ndarray): image to cut
        mask (numpy.ndarray): mask of image to use for cutting

    Returns:
        cut_image (numpy.ndarray): cut image    
    '''

    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_image = image[row_min:row_max+1, col_min:col_max+1]

    return cut_image

def midpointGroup9(image):
    '''Find midpoint of image array.'''
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid

def asymmetry(mask):
    '''Calculate asymmetry score between 0 and 1 from vertical and horizontal axis
    on a binary mask, 0 being complete symmetry, 1 being complete asymmetry, 
    i.e. no pixels overlapping when folding mask on x- and y-axis

    Args:
        mask (numpy.ndarray): input mask
    
    Returns:
        asymmetry_score (float): Float between 0 and 1 indicating level of asymmetry.    
    '''

    row_mid, col_mid = midpointGroup9(mask)

    # Split mask into halves hortizontally and vertically
    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    # Flip one half for each axis
    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    # Use logical xor to find pixels where only one half is present
    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    # Compute sums of total pixels and pixels in asymmetry areas
    total_pxls = np.sum(mask)
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    # Calculate asymmetry score
    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)

def rotation_asymmetry(mask, n: int):
    '''Rotate mask n times and calculate asymmetry score for each iteration.
    Rotates n times between 0 and 90 degrees, as 90 degree rotations do not change the
    asymmetry score, i.e., a 30 degree rotation is the same as a 120 degree rotation.

    Args:
        mask (numpy.ndarray): input mask  
        n (int): amount of rotations

    Returns:
        asymmetry_scores (dict): dict of asymmetry scores calculated from each rotation.    
    '''
    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees)
        cutted_mask = cut_mask(rotated_mask)

        asymmetry_scores[degrees] = asymmetry(cutted_mask)

    return asymmetry_scores

def mean_asymmetry(mask, rotations = 30):
    '''Return mean asymmetry score from mask.
    Optional argument (defualt 30) rotations decides amount of rotations in asymmetry calculation
    
    Args:
        mask (numpy.ndarray): mask to compute asymmetry score for
        rotations (int, optional): amount of rotations (default 30)

    Returns:
        mean_score (float): mean asymmetry score.
    '''
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score          

def best_asymmetry(mask, rotations = 30):
    '''Return best (lowest) asymmetry score from mask.
    Optional argument (defualt 30) rotations decides amount of rotations in asymmetry calculation
    
    Args:
        mask (numpy.ndarray): mask to compute asymmetry score for
        rotations (int, optional): amount of rotations (defualt 30)

    Returns:
        best_score (float): best asymmetry score.
    '''
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    best_score = min(asymmetry_scores.values())

    return best_score

def worst_asymmetry(mask, rotations = 30):
    '''Return worst (highest) asymmetry score from mask.
    Optional argument (defualt 30) rotations decides amount of rotations in asymmetry calculation
    
    Args:
        mask (numpy.ndarray): mask to compute asymmetry score for
        rotations (int, optional): amount of rotations (defualt 30)

    Returns:
        worst_score (float): worst asymmetry score.
    '''
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    worst_score = max(asymmetry_scores.values())

    return worst_score  

def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
    '''Get color segments of lesion from SLIC algorithm. 
    Optional argument n_segments (defualt 50) defines desired amount of segments.
    Optional argument compactness (defualt 0.1) defines balance between color 
    and position.

    Args:
        image (numpy.ndarray): image to segment
        mask (numpy.ndarray):  image mask
        n_segments (int, optional): desired amount of segments
        compactness (float, optional): compactness score, decides balance between
            color and and position

    Returns:
        slic_segments (numpy.ndarray): SLIC color segments.
    '''
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)
    
    return slic_segments

def get_rgb_means(image, slic_segments):
    '''Get mean RGB values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        rgb_means (list): RGB mean values for each segment.
    '''

    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(1, max_segment_id + 1):

        #Create masked image where only specific segment is active
        segment = image.copy()
        segment[slic_segments != i] = -1

        #Get average RGB values from segment
        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))
        
        rgb_means.append(rgb_mean) 
        
    return rgb_means

def get_hsv_means(image, slic_segments):
    '''Get mean HSV values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        hsv_means (list): HSV mean values for each segment.
    '''

    hsv_image = rgb2hsv(image)

    max_segment_id = np.unique(slic_segments)[-1]

    hsv_means = []
    for i in range(1, max_segment_id + 1):

        # Create masked image where only specific segment is active
        segment = hsv_image.copy()
        segment[slic_segments != i] = nan

        #Get average HSV values from segment
        hue_mean = circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit') # Compute circular hue mean
        sat_mean = np.mean(segment[:, :, 1], where = (slic_segments == i)) # Compute saturation mean
        val_mean = np.mean(segment[:, :, 2], where = (slic_segments == i)) # Compute value mean

        hsv_mean = np.asarray([hue_mean, sat_mean, val_mean])

        hsv_means.append(hsv_mean)
        
    return hsv_means

def rgb_var(image, slic_segments):
    '''Get variance of RGB means for each segment in 
    SLIC segmentation in red, green and blue channels

    Args:
        image (numpy.ndarray): image to compute color variance for
        slic_segments (numpy.ndarray): array containing SLIC segmentation
    
    Returns:
        red_var (float): variance in red channel segment means
        green_var (float): variance in green channel segment means
        blue_var (float): variance in green channel segment means.
    '''

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2: # Use 2 since slic_segments also has 0 for area outside mask
        return 0, 0, 0

    rgb_means = get_rgb_means(image, slic_segments)
    n = len(rgb_means) # Amount of segments, used later to compute variance

    # Seperate and collect channel means together in lists
    red = []
    green = []
    blue = []
    for rgb_mean in rgb_means:
        red.append(rgb_mean[0])
        green.append(rgb_mean[1])
        blue.append(rgb_mean[2])

    # Compute variance for each channel seperately
    red_var = variance(red, sum(red)/n)
    green_var = variance(green, sum(green)/n)
    blue_var = variance(blue, sum(blue)/n)

    return red_var, green_var, blue_var

def hsv_var(image, slic_segments):
    '''Get variance of HSV means for each segment in 
    SLIC segmentation in hue, saturation and value channels

    Args:
        image (numpy.ndarray): image to compute color variance for
        slic_segments (numpy.ndarray): array containing SLIC segmentation
    
    Returns:
        hue_var (float): variance in hue channel segment means
        sat_var (float): variance in saturation channel segment means
        val_var (float): variance in value channel segment means.
    '''

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2: # Use 2 since slic_segments also has 0 marking for area outside mask
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments)
    n = len(hsv_means) # Amount of segments, used later to compute variance

    # Seperate and collect channel means together in lists
    hue = []
    sat = []
    val = []
    for hsv_mean in hsv_means:
        hue.append(hsv_mean[0])
        sat.append(hsv_mean[1])
        val.append(hsv_mean[2])

    # Compute variance for each channel seperately
    hue_var = circvar(hue, high=1, low=0)
    sat_var = variance(sat, sum(sat)/n)
    val_var = variance(val, sum(val)/n)

    return hue_var, sat_var, val_var


def color_dominance(image, mask, clusters = 5, include_ratios = False):
    '''Get the most dominent colors of the cut image that closest sorrounds the lesion using KMeans

    Args:
        image (numpy.ndarray): image to compute dominent colors for
        mask (numpy.ndarray): mask of lesion
        clusters (int, optional): amound of clusters and therefore dominent colors (defualt 3)
        include_ratios (bool, optional): whether to include the domination ratios for each color (defualt False)

    Return:  
        if include_ratios == True: 
            p_and_c (list): list of tuples, each containing the percentage and RGB array of the dominent color
        else: 
            dom_colors (array): array of RGB arrays of each dominent color.
    '''
    
    # Prepare image for KMeans
    cut_im = cut_im_by_mask(image, mask) # Cut image to remove excess skin pixels
    hsv_im = rgb2hsv(cut_im) # Convert image to HSV color space
    flat_im = np.reshape(hsv_im, (-1, 3)) # Flatten image to 2D array

    # Use KMeans to cluster image by colors
    k_means = KMeans(n_clusters=clusters, n_init=10, random_state=0)
    k_means.fit(flat_im)

    # Save cluster centers (dominant colors) in array
    dom_colors = np.array(k_means.cluster_centers_, dtype='float32') 

    if include_ratios:

        counts = np.unique(k_means.labels_, return_counts=True)[1] # Get count of each dominent color
        ratios = counts / flat_im.shape[0] # Get percentage of total image for each dominent color

        r_and_c = zip(ratios, dom_colors) # Percentage and colors
        r_and_c = sorted(r_and_c, key=lambda x: x[0],reverse=True) # Sort in descending order

        return r_and_c
    
    return dom_colors

def compactness_score(mask):
    '''Computes a compactness score for the given mask.
    The score is based of the Polsby-Popper measure. 
    The score falls between the value 0 and 1. Scores closer to 1 indicates a more compact mask.

    Args:
        mask (numpy.ndarray): input masked image
    
    Returns:
        compactness_score (float): Float between 0 and 1 indicating compactness. 
    '''

     #Area of ground truth 
    A = np.sum(mask)

    #Structural element, that we will use as a "brush" on our mask
    struct_el = morphology.disk(2)

    # Use this "brush" to erode the image - eat away at the borders
    mask_eroded = morphology.binary_erosion(mask, struct_el)

    #Finding the perimeter of the ground truth
    perimeter = mask - mask_eroded

    #Length of the perimeter
    l = np.sum(perimeter)

    compactness = (4*pi*A)/(l**2)

    score = round(1-compactness, 3)

    return compactness

def convexity_score(mask):
    '''Calculate convexity score between 0 and 1, 
    with 0 indicating a smoother border and 1 a more crooked border.

    Args:
        image (numpy.ndarray): input masked image
    
    Returns:
        convexity_score (float): Float between 0 and 1 indicating convexity.    
    '''

    # Get coordinates of all pixels in the lesion mask
    coords = np.transpose(np.nonzero(mask))

    # Compute convex hull of lesion pixels
    hull = ConvexHull(coords)

    # Compute area of lesion mask
    lesion_area = np.count_nonzero(mask)

    # Compute area of convex hull
    convex_hull_area = hull.volume + hull.area

    # Compute convexity as ratio of lesion area to convex hull
    convexity = lesion_area / convex_hull_area
    
    return convexity #round(1-convexity, 3)

def get_relative_rgb_means(image, slic_segments):
    '''Get mean RGB values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        rgb_means (list): RGB mean values for each segment.
    '''

    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(0, max_segment_id + 1):

        #Create masked image where only specific segment is active
        segment = image.copy()
        segment[slic_segments != i] = -1

        #Get average RGB values from segment
        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))
        
        rgb_means.append(rgb_mean) 

    rgb_means_lesion = np.mean(rgb_means[1:],axis=0)
    rgb_means_skin = np.mean(rgb_means[0])

    F1, F2, F3 = rgb_means_lesion/sum(rgb_means_lesion)
    F10, F11, F12 = rgb_means_lesion - rgb_means_skin
        
    return F1, F2, F3, F10, F11, F12