# A Comparative Study of Feature Robustness and Sex Bias in Logistic Regression and CNN for Skin Cancer Detection

This repository contains the code produced in the making of the bachelor project [A Comparative Study of Feature Robustness and Sex Bias in Logistic Regression and CNN for Skin Cancer Detection]() INSERT LINK TO PDF IN GIT REPO, made by Nikolette Pedersen, Regitze Sydendal & Andreas Wulff.

Included are the implementations of:
- Extracting features from images
- Augmenting images
- Spliting data into test/val/training
- Training a logistic regression and convultional neural network
- Providing data and plots

## Installation

The images and metadata from the Pad-Ufes-20 dataset are required to recreate our experiments. Furthermore are the manually created segmentations corresponding to the images needed. These are both provided in this git repo, as the Pad-Ufes-20 is manually modified. To read more, look at the attached paper. 

Libraries can be instaled using the package manager [pip](https://pip.pypa.io/en/stable/). 
Libraries needed to reproduce our experiments: [^1]
[^1]: Disclaimer: It's possible that this list is not exhaustive
### Standard libraries for data manipulation and OS operations
- os
- shutil
- numpy
- pandas

# Visualization libraries
- matplotlib.pyplot
- seaborn

# Image processing and machine learning libraries
- PIL
- cv2
- skimage
- sklean
- scipy
- statistics

# Statistical modeling and metrics
- statsmodels

# PyTorch for deep learning
- torch
- torchvision

# MLflow for model tracking
- mlflow



## Acknowledgements

## License

[MIT](https://choosealicense.com/licenses/mit/)
