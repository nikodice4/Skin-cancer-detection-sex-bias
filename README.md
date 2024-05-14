# A Comparative Study of Feature Robustness and Sex Bias in Logistic Regression and CNN for Skin Cancer Detection

This repository contains the code produced in the making of the bachelor project [A Comparative Study of Feature Robustness and Sex Bias in Logistic Regression and CNN for Skin Cancer Detection]() INSERT LINK TO PDF IN GIT REPO, made by Nikolette Pedersen, Regitze Sydendal & Andreas Wulff.

Included are the implementations of:
- Extracting features from images
- Augmenting images
- Spliting data into test/val/training
- Training a logistic regression and convultional neural network
- Providing statistical analysis of the results

Also included are:
- Images and metadata from PAD-ufes-20, as well as masks and metadata fixed.
- All of the extracted features 
- Results
- A script to quickly recreate our results

## Installation

The images and metadata from the Pad-Ufes-20 dataset are required to recreate our experiments. Furthermore are the manually created segmentations corresponding to the images needed. These are both provided in this git repo, as the Pad-Ufes-20 is manually modified. To read more, look at the attached paper. 

Python is needed.
Libraries can be instaled using the package manager [pip](https://pip.pypa.io/en/stable/). 
Libraries needed to reproduce our experiments: [^1]
[^1]: Disclaimer: It's possible that this list is not exhaustive
### Standard libraries for data manipulation and OS operations
- os
- shutil
- numpy
- pandas (v. 2.2.0)[^2]
[^2]: Pandas might be version-sensitive
### Visualization libraries
- matplotlib.pyplot
- seaborn

### Image processing and machine learning libraries
- PIL
- cv2
- skimage
- sklean
- scipy
- statistics

### Statistical modeling and metrics
- statsmodels

### PyTorch for deep learning
- torch
- torchvision

### MLflow for model tracking
- mlflow

## Recreation

In order to quickly recreate some of our results, a script is provided in the 'recreation' directory. This will output data used in our tables to a txt file, stored in 'recreation/table_values', and figures as png-files to 'recreation/figures'. It is recommended to run this script with the data already provided, or after running the ML-models yourself.

In order to recreate all of our experiments from scratch, only the data in 'data/images/lesion_images', 'data/images/lesion_masks' and 'data/metadata/fixed_metadata' are required. These are either manually created, or provided by Pad-Ufes-20 and then manually corrected by us.

### 0: Data exploration

Various curiosities and data we have had a need for can be extracting using the various scripts and notebooks in 'exploration'. This is not needed to recreate the experiments, but could be of interest.

### 1: Create additional data metadata and split data for ML models [^3]
[^3]: Note that this is computionally heavy and not recommended, when everything already is in this repository
1. Run everything in 'split/cnn_split.ipynb'. This will create the augmented images, metadata for these. Furthermore it will split csv files into two directories 'data/cnn/cnn_splitted_data_once_augmented' and 'data/cnn/cnn_splitted_data_50_50_split'. The difference between these are how much data is augmented. We have elected to proceed exclusively with once augmented, but in future works it could be interesting to experiment with 50/50. 
2. Now run everything in 'split/feature_extractions.ipynb'. This will create metadata files with all features for both the *clean* metadata and the one with augmentations.
3. Split the data for the logistic regression by running everything in 'split/lr_split_aug.ipynb'.
### 2: Run ML models
- **CNN:** Navigate to 'ml_models/cnn/cnn_run.py'. Before running, change to port of your choice at the bottom of the script. Then start mlflow by running the following command in your terminal:
```bash
mlflow ui -p <insert-port>
```
Then run the script.
- **LR:** Do the same as above but in 'ml_models/lr/lr_run.py'.

### 3: Analyse data
- Run everything in 'analysis/plots/plot_cnn_regression.ipynb'
- Run everything in 'analysis/plots/plot_lr_regression.ipynb'
- Run everything in 'analysis/plots/significant_testing.ipynb'
- Outputs can be found collected in 'analysis/plots/cm_plots', 'analysis/plots/cnn_plots' & 'analysis/plots/lr_plots'


## Acknowledgements

- Dataset provided by [Pad-Ufes-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) [CC BY License](https://creativecommons.org/licenses/by/4.0/). Certain manual changes have been made by us to the dataset.
- Inspiration for project, datasplit and more provided by Petersen et al., MICCAI 2022 [Feature robustness and sex differences in medical imaging: a case study in MRI-based Alzheimer's disease detection](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_9). [Github](https://github.com/e-pet/adni-bias).
- Feature extraction was gathered from anonymized first-year students from the IT-University of Copenhagen. Contact Veronika Cheplygina for further enquiries.
## License

[MIT](https://choosealicense.com/licenses/mit/)
