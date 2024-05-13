# code for resnet model
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import multiprocessing

# import lightning libs
import pytorch_lightning as pl
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn
import pandas as pd
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from torchmetrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
import os

from datetime import datetime
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datetime import datetime
import argparse
import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom dataset for loading images and labels which also applies transforms
    """
    def __init__(self, dataframe, image_folder, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_column = "label"

        if image_folder[-1] != "/":
            image_folder += "/"
        self.image_folder = image_folder

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_name"]
        image = Image.open(self.image_folder+img_path)
        # convert image to tensor
        image = transforms.ToTensor()(image)
        
        label = float(self.dataframe[self.label_column][idx])

        # Move tensors to the GPU
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, image_folder, batch_size=32, model_name="resnet"):
        super().__init__()
        self.model_name = model_name
        self.image_folder = image_folder
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.num_workers = multiprocessing.cpu_count()
        # self.num_workers = 2

    def setup(self, stage=None):
        if self.train_df is not None:
            self.train_dataset = CustomDataset(self.train_df, transform=self._transforms(),image_folder=self.image_folder)
        self.val_dataset = CustomDataset(self.val_df, transform=self._transforms(), image_folder=self.image_folder)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def _transforms(self):
        if self.model_name == "resnet":
            return transforms.Compose([
                ResNet50_Weights.DEFAULT.transforms()
            ])
        else: 
            print("Model transforms not recognized")
            return exit()

class ResNet(nn.Module):
    """
    ResNet is based on ResNet50
    Change num_channels to 3 if using RGB images
    """
    def __init__(self, num_channels=3, second_last_layer_size=256, last_layer_size=512):
        self.num_channels = num_channels
        self.second_last_layer_size = second_last_layer_size
        self.last_layer_size = last_layer_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(ResNet, self).__init__()
    
        resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Modify the first convolutional layer to accept one channel instead of three
        resnet_model.conv1 = torch.nn.Conv2d(self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        for param in resnet_model.parameters():
            param.requires_grad = False

        resnet_model.fc = nn.Linear(2048, self.second_last_layer_size)

        output_head = nn.Sequential(
            nn.Linear(self.second_last_layer_size, self.last_layer_size),
            nn.ReLU(),
            nn.Linear(self.last_layer_size, 1),
            nn.Sigmoid()
        )

        self.base_model = resnet_model.to(self.device)
        self.output_head = output_head.to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        base_prediction = self.base_model(x)
        output_prediction = self.output_head(base_prediction)
        return output_prediction
    


load_dotenv()

class LitModel(pl.LightningModule):
    def __init__(self, model, model_name, train_files, train_csv, test_files, test_csv, image_folder, learning_rate=1e-5, batch_size=128, second_last_layer_size=1024, last_layer_size=128):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.train_files = train_files
        self.train_csv = train_csv
        self.test_files = test_files
        self.test_csv = test_csv
        self.image_folder = image_folder  # Add image_folder attribute
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.second_last_layer_size = second_last_layer_size
        self.last_layer_size = last_layer_size
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.auc = BinaryAUROC() 
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.prepare_data()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        logits = logits.squeeze()
        loss = self.loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        logits = logits.squeeze()
        loss = self.loss(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(logits, y))
        self.log('val_auc', self.auc(logits, y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        logits = logits.squeeze()
        loss = self.loss(logits, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.accuracy(logits, y))
        self.log('test_auc', self.auc(logits, y))

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    
    def train_dataloader(self):
        return self.train_data_module.train_dataloader()
    
    def val_dataloader(self):
        return self.train_data_module.val_dataloader()

    def test_dataloader(self):
        return self.test_data_module.test_dataloader()

def prepare_data(self):
    train_data = pd.read_csv(self.train_csv)
    test_data = pd.read_csv(self.test_csv)

    train_df, val_df = train_test_split(train_data, test_size=0.1, random_state=42)
    test_df = test_data

    self.train_data_module = CustomDataModule(train_df=train_df, val_df=val_df, batch_size=self.batch_size, image_folder=self.image_folder, model_name=self.model_name)
    self.train_data_module.setup()

    self.test_data_module = CustomDataModule(train_df=None, test_df=test_df, batch_size=self.batch_size, image_folder=self.image_folder, model_name=self.model_name)
    self.test_data_module.setup()



    def train_dataloader(self):
        return self.train_data_module.train_dataloader()
    
    def val_dataloader(self):
        return self.train_data_module.val_dataloader()

    def test_dataloader(self):
        return self.test_data_module.test_dataloader()

   
def run_model(model_name, batch_size, learning_rate, max_epochs, image_folder, log_name):
    log_name = f"{datetime.now().strftime('%Y-%m-%d')}-{log_name}"
    if model_name == "resnet":
        model = ResNet()
    else:
        raise ValueError("Model not recognized")

    lit_model = LitModel(model, model_name, train_files=args.train_csv_file, train_csv=args.train_csv_file, test_files=args.test_csv_file, test_csv=args.test_csv_file, image_folder=image_folder, batch_size=batch_size, learning_rate=learning_rate)
    lit_model.prepare_data()

    logger = TensorBoardLogger('lightning_logs', name=log_name)

    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        logger=logger, 
        devices=1,
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=True, 
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True)
            ]
        )
    
    trainer.fit(lit_model, lit_model.train_dataloader(), lit_model.val_dataloader())
    trainer.test(lit_model, lit_model.test_dataloader())

    # save model
    print(f"Model saved under: Resnet50/models/{log_name}.ckpt")
    trainer.save_checkpoint(f"ResNet50/models/{log_name}.ckpt")



def split_images(train_csv_file, test_csv_file, image_folder):
    # Load CSV files
    train_data = pd.read_csv(train_csv_file)
    test_data = pd.read_csv(test_csv_file)
    
    # Extract image IDs
    train_img_ids = train_data['img_id'].tolist()
    test_img_ids = test_data['img_id'].tolist()
    
    # Get list of all image filenames
    image_files = os.listdir(image_folder)
    
    # Split image filenames into train and test sets based on IDs
    train_files = [filename for filename in image_files if filename.split('.')[0] in train_img_ids]
    test_files = [filename for filename in image_files if filename.split('.')[0] in test_img_ids]
    
    return train_files, test_files


def main():
    train_files, test_files = split_images(args.train_csv_file, args.test_csv_file, args.image_folder)
    run_model(args.model, args.batch_size, args.learning_rate, args.max_epochs, args.image_folder, args.model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model using PyTorch Lightning.')
    parser.add_argument('--image_folder', default="pad-ufes/images", type=str, help='Path to folder containing all images')
    parser.add_argument('--train_csv_file', default="data/splitted_csv/m_f_ca_nc_train_0_0.00_0.csv", type=str, help='Path to CSV file containing image labels')
    parser.add_argument('--test_csv_file', default="data/splitted_csv/m_f_ca_nc_test_0.csv", type=str, help='Path to CSV file containing validation image labels')
    parser.add_argument('--model', type=str, default='resnet', help='Model to train (resnet, vit, cnn)')
    parser.add_argument('--learning_rate', type=float, default=1.7446772512668928e-05, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=50, help='Number of epochs to train for')
    args = parser.parse_args()

    
    main()    