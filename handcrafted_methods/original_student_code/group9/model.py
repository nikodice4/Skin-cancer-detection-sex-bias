### Main model module. This module contains functions used in the trainer
### script to build and train the standardizing scaler, PCA, feature selection
### and classifier algorithms, as well as functions to extract features, process images
### and evaluate features.
### Apply functions are used in notebooks to evaluate the models, while the main
### classify script (classify.py) uses modifies versions.

###############
### IMPORTS ###
###############

# Standard Modules
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, make_scorer 

# Feature extraction (PCA)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Feature selection
from sklearn.feature_selection import SelectKBest,mutual_info_classif

# For loading and saving classifiers
import pickle as pk

# Image preperation
from prep_image import prep_im_and_mask

# Feature extraction
from asymmetry import mean_asymmetry, best_asymmetry, worst_asymmetry
from color import slic_segmentation, rgb_var, hsv_var, color_dominance, get_relative_rgb_means
from compactness import compactness_score
from convexity import convexity_score

###########################
### FEATURE EXTRACTIONS ###
###########################

def extract_features(im, im_mask):
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
	mean_asym = mean_asymmetry(im_mask,4)
	best_asym = best_asymmetry(im_mask,4)
	worst_asym = worst_asymmetry(im_mask,4)

	# Color variance
	segments = slic_segmentation(im, im_mask, n_segments=250)
	red_var, green_var, blue_var = rgb_var(im, segments)
	hue_var, sat_var, val_var = hsv_var(im, segments)

	# Color dominance
	dom_colors = color_dominance(im, im_mask, clusters=5, include_ratios=True) # Extract five most dominent colors, sorted by percentage of total area
	dom_hue, dom_sat, dom_val = dom_colors[0][1]     

	# Compactness
	compactness = compactness_score(im_mask)

	# Convexity
	convexity = convexity_score(im_mask)

	# Relative color scores
	F1, F2, F3, F10, F11, F12 = get_relative_rgb_means(im, segments)

	return [mean_asym, best_asym, worst_asym, red_var, green_var, \
		blue_var, hue_var, sat_var, val_var, dom_hue, dom_sat, \
		dom_val, compactness, convexity, F1, F2, F3, F10, F11, F12]

########################
### IMAGE PROCESSING ###
########################

def ProcessImages(file_data, image_folder, mask_folder, file_features, feature_names):
    '''
    Process images and extract features.
    
    Args:
        file_data (str): File path or file object containing metadata.
        image_folder (str): Path to the folder containing the images.
        mask_folder (str): Path to the folder containing the masks.
        file_features (str): File path or file object to save the extracted features.
        feature_names (list): List of feature names.
    '''
    # Import metadata from file
    df = pd.read_csv(file_data)

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
            print(f"Image or mask for {id} not found, skipping...")

            continue

        try:
            im, mask = prep_im_and_mask(id, image_folder, mask_folder)
            x = extract_features(im, mask)
            img_ids.append(id)
            features.append(x)  # Append the features for each image
            print(f"Processed {i+1} out of {len(df)} images")
        except Exception as e:
            print(f"Error processing {id_with_extension}: {e}, skipping...")
            continue

    # Check if we have processed any images
    if img_ids:
        # Convert the list of features to a numpy array
        features_array = np.array(features, dtype=np.float16)
        
        # Insert image ids
        df_features = pd.DataFrame(features_array, columns=feature_names)
        df_features.insert(0, 'ID', img_ids)

        # Save feature data as csv file
        df_features.to_csv(file_features, index=False)
        print("Feature extraction complete.")
    else:
        print("No images were processed.")


###########
### PCA ###
###########

def train_scaler(X):
	'''Train the standardizing scaler. Save the result.
	
	Args:
		X (pandas.DataFrame): Data Frame of features.
	'''

	std_scl = StandardScaler().set_output(transform='pandas')
	std_scl.fit_transform(X)

	pk.dump(std_scl, open('scaler.pkl', 'wb'))	

# Standardize feature data
def apply_scalar(X):
	'''Standardize the input.

    Args:
        X (pandas.DataFrame): Data Frame of features.

    Returns:
        X_std (pandas.DataFrame): Array containg standardized features.    
    '''

	scaler = pk.load(open('scaler.pkl', 'rb'))
	X_scaled = scaler.transform(X)

	return X_scaled

def train_pca(X, n=0.95):
	'''Train PCA. Save the result.

	Args:
	    X (pandas.DataFrame): Data Frame of features.
	    n (float): Percentage of variation that should be explained by the chosen features.
	'''

	# X_normalized = (X - X.mean()) / X.std()
	pca = PCA(n_components=n)
	pca.fit_transform(X)

	pk.dump(pca, open('pca.pkl', 'wb'))

def apply_pca(X):
	'''Apply pca to X.

	Args:
	    X (pandas.DataFrame): Data Frame of features.
	    n (float): Percentage of variation that should be explained by the chosen features.

	Returns:
	    X_std_pca (numpy.ndarray): Array containg transformed, standardized features.    
	'''

	pca = pk.load(open('pca.pkl', 'rb'))
	X_transformed = pca.transform(X)

	return X_transformed

#########################
### FEATURE SELECTION ###
#########################

def train_feature_selector(X_train, y_train, k):
    '''Using SelectKBest to extract features from train_X, down to k features as output
    Returns a selector object (which is applied to X_train and X_test afterwards) 
	and the score for each feature.

    Args:
        train_X (pandas.DataFrame): Data Frame of features from X_train.
		train_y (pandas.DataFrame): Data Frame of target values from y_train.
        k (int): Number of features to output.

    Returns:
		feature_selector (selector object): 
        scores (numpy.ndarray): Array containg scores for each feature.    
    '''

    feature_selector = SelectKBest(mutual_info_classif, k=k)
    feature_selector.fit_transform(X_train, y_train)
    
    pk.dump(feature_selector, open('selector.pkl', 'wb'))
    
    return feature_selector

def apply_feature_selector(X):
	'''Apply feature selector to transform the input features.
    
    Args:
        X (array-like): Input feature matrix or dataset.
        
    Returns:
        array-like: Transformed feature matrix.
    '''

	feature_selector = pk.load(open('selector.pkl', 'rb'))
	X_transformed = feature_selector.transform(X)

	return X_transformed

def feature_scores(feature_selector):
    '''Get feature scores from feature selector.'''
    
    scores = feature_selector.scores_
    
    return scores

########################
### TRAIN CLASSIFIER ###
########################

def train_clf(X_train, y_train, classifiers):
	'''
    Train multiple classifiers on the input training data.

    Args:
        X_train (array-like): Input feature matrix or dataset for training.
        y_train (array-like): Target variable or labels for training.
        classifiers (list): List of classifier objects to be trained.

    Returns:
        list: List of trained classifier objects.
    '''

	trained_classifiers = [classifier.fit(X_train, y_train) for classifier in classifiers]

	return trained_classifiers

def cross_validate_clf(X, y, classifiers, groups):
	'''Perform cross-validation for multiple classifiers on the input data.

    Args:
        X (array-like): Input feature matrix or dataset.
        y (array-like): Target variable or labels.
        classifiers (list): List of classifier objects to be evaluated.
        groups (array-like): Groups or categories for stratified grouping in cross-validation.

    Returns:
        dict: Dictionary containing evaluation results for each classifier, with metrics such as Accuracy, Sensitivity,
              Specificity, Precision, and ROC AUC.
	'''

	# Scores for evaluation
	scores ={'accuracy': make_scorer(accuracy_score), 
	  'sensitivity': make_scorer(recall_score), 
	  'specificity': make_scorer(recall_score, pos_label=0), 
	  'precision': make_scorer(precision_score), 
	  'roc_auc': make_scorer(roc_auc_score, needs_proba=True)}

	num_folds = 5
	cross_val = StratifiedGroupKFold(n_splits= num_folds)	

	evaluation_results = {}
	for classifier in classifiers:
		cv_results = cross_validate(classifier, X, y, scoring=scores, cv=cross_val, groups = groups)
		
		if type(classifier).__name__ == "KNeighborsClassifier":
			classifier_name = type(classifier).__name__
			params_dict = classifier.get_params()
			n_neigbors = params_dict["n_neighbors"]
			classifier_name = f"{classifier_name} with n_neighbors={n_neigbors}"
		else:
			classifier_name = type(classifier).__name__

		evaluation_results[classifier_name] = {
            'Accuracy': cv_results['test_accuracy'].mean(),
            'Sensitivity': cv_results['test_sensitivity'].mean(),
            'Specificity': cv_results['test_specificity'].mean(),
            'Precision': cv_results['test_precision'].mean(),
            'ROC AUC': cv_results['test_roc_auc'].mean()

        }

	return evaluation_results

def evaluate_clf(X_test, y_test, trained_classifiers):
	'''
    Evaluate the performance of trained classifiers on the test data.

    Args:
        X_test (array-like): Input feature matrix or dataset for testing.
        y_test (array-like): Target variable or labels for testing.
        trained_classifiers (list): List of trained classifier objects.

    Returns:
        dict: Dictionary containing evaluation results for each classifier, with metrics such as Accuracy, Sensitivity,
              Specificity, Precision, and AUC ROC.
    '''
    
	# Take trained classifiers as inputs
	results = {}
	for clf in trained_classifiers:
		y_pred = clf.predict(X_test)

		if type(clf).__name__ == "KNeighborsClassifier":
			classifier_name = type(clf).__name__
			params_dict = clf.get_params()
			n_neigbors = params_dict["n_neighbors"]
			classifier_name = f"{classifier_name} with n_neighbors: {n_neigbors}"
		else:
			classifier_name = type(clf).__name__
  
		results[classifier_name] = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 3),
            'Sensitivity': round(recall_score(y_test, y_pred, pos_label = 1), 3),
            'Specificity': round(recall_score(y_test, y_pred, pos_label = 0),3),
            'Precision': round(precision_score(y_test, y_pred), 3),
        	'AUC ROC': round(roc_auc_score(y_test, y_pred), 3)
        }
    
	return results

def print_results(results):
	'''
    Print the evaluation results.

    Args:
        results (dict): Dictionary containing evaluation results for each classifier.
    '''
	for classifier, scores in results.items():
		print(classifier)
		for metric, score in scores.items():
			print(f'{metric}: {score:.4f}')
	    
		print()

def print_results_cv(evaluation_results):
	'''Print the evaluation results for cross-validated classifiers.

    Args:
        evaluation_results (dict): Dictionary containing evaluation results for each classifier.
    '''

	for classifier, scores in evaluation_results.items():
		print(classifier)
		for metric, score in scores.items():
			print(f'{metric}: {score:.4f}')
	    
		print()
