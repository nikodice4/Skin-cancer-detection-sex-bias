# Imports
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

# Creaitng a CustomDataset class so the DataLoader can read it
class CustomDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, img_id_column="img_id"):
        self.data_dir = data_dir
        self.transform = transform
        self.img_id_column = img_id_column
        self.data = pd.read_csv(csv_file)

    # We get the lenght of the data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Here we get the image and the label from the dataset
        img_name = os.path.join(self.data_dir, self.data.iloc[idx, self.data.columns.get_loc(self.img_id_column)])
        if not os.path.exists(img_name):
            img_name = os.path.join("data/images/augmented_images_once",
                                    self.data.iloc[idx, self.data.columns.get_loc(self.img_id_column)])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
 
        diagnostic = self.data.iloc[idx]["diagnostic"]
        label = 1 if diagnostic in ["BCC", "MEL", "SCC"] else 0  # 1 for cancerous, 0 for non-cancerous
    
        return image, label
    
# We do the same but for females separately
class FemaleCustomDataset(CustomDataset):
    def __init__(self, csv_file, data_dir, transform=None, img_id_column="img_id", gender_column="gender"):
        super().__init__(csv_file, data_dir, transform, img_id_column)
        self.gender_column = gender_column
        self.data = self.data[self.data[gender_column] == "FEMALE"]

# And for males separately
class MaleCustomDataset(CustomDataset):
    def __init__(self, csv_file, data_dir, transform=None, img_id_column="img_id", gender_column="gender"):
        super().__init__(csv_file, data_dir, transform, img_id_column)
        self.gender_column = gender_column
        self.data = self.data[self.data[gender_column] == "MALE"]