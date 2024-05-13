import os
from model import *
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#################
### COMSTANTS ###
#################

file_data = '/Users/regitzesydendal/Documents/itu/6 semester/bachelor/data/new_padufes/metadata.csv'
image_folder = '/Users/regitzesydendal/Documents/itu/6 semester/bachelor/data/new_padufes/imgs_part_1/'
mask_folder = '/Users/regitzesydendal/Documents/itu/6 semester/bachelor/data/new_padufes/new_padufes_masks/'
file_features = '/Users/regitzesydendal/Documents/itu/6 semester/bachelor/data/feature_group9_new.csv'

feature_names = ['mean_asymmetry', 'best_asymmetry', 'worst_asymmetry', 'red_var', 'green_var', \
     'blue_var', 'hue_var', 'sat_var', 'val_var', 'dom_hue', 'dom_sat', 'dom_val', \
     'compactness', 'convexity', 'F1', 'F2', 'F3', 'F10', 'F11', 'F12']

ProcessImages(file_data, image_folder, mask_folder, file_features, feature_names)