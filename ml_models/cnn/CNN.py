# Imports
import torch.nn as nn
import torch.optim as optim
import torchvision
print(torchvision.__version__)
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import match_id_to_image as id2img
import torch
torch.manual_seed(42)


def resnet50(train_csv, val_csv, test_csv, img_directory):
    # We load in the weights for the ResNet-50
    weighted_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # We moditfy the last layer to have one output neuron for the binary classification task
    num_ftrs = weighted_model.fc.in_features
    weighted_model.fc = nn.Linear(num_ftrs, 1)

    # Here we prepare the dataset, by resizing the images to 224x224, converting them to tensors and normalizing them
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Creating the Datasets using 
    train_dataset = id2img.CustomDataset(csv_file=train_csv, data_dir=img_directory, transform=transform)
    val_dataset = id2img.CustomDataset(csv_file=val_csv, data_dir=img_directory, transform=transform)
    test_dataset = id2img.CustomDataset(csv_file=test_csv, data_dir=img_directory, transform=transform)
    test_dataset_FEMALE = id2img.FemaleCustomDataset(csv_file=test_csv, data_dir=img_directory, transform=transform)
    test_dataset_MALE = id2img.MaleCustomDataset(csv_file=test_csv, data_dir=img_directory, transform=transform)

    # Creating the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=104)
    test_loader_female = DataLoader(test_dataset_FEMALE, batch_size=104)
    test_loader_male = DataLoader(test_dataset_MALE, batch_size=104)

    # Here we define our loss function and our optimiser
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(weighted_model.parameters(), lr=0.001)

    # Here we train the model
    # We squeeze the outputs to match the shape of the labels, and ensure that the labels are floats
    num_epochs = 10
    for epoch in range(num_epochs):
        weighted_model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = weighted_model(inputs)
            outputs = outputs.squeeze(1) 
            loss = criterion(outputs, labels.float())  
            loss.backward()
            optimizer.step()

        # Here we validate the model
        weighted_model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_outputs = weighted_model(val_inputs)
                val_outputs = val_outputs.squeeze(1)
                val_loss += criterion(val_outputs, val_labels.float()).item()
        print(f"Epoch {epoch+1}, Validation loss: {val_loss / len(val_loader)}")

    # Then we evaluate on the test set after the training is complete
    weighted_model.eval()
    with torch.no_grad():
        # Assume evaluate_accuracy is a function you define to calculate accuracy
        test_accuracy = evaluate_accuracy(test_loader, weighted_model)
        print(f"Test accuracy: {test_accuracy}")

    return test_loader, test_loader_female, test_loader_male, val_loader, weighted_model


# This is a helper function to evaluate accuracy
# We convert the logits to probabilities, and then convert the probabilities to 0 or 1 based on a threshold, to get the output
def evaluate_accuracy(data_loader, model, threshold=0.5):
    correct = 0
    total = 0
    for images, labels in data_loader:
        outputs = model(images)
        outputs = torch.sigmoid(outputs)  # Convert logits to probabilities
        predicted = (outputs > threshold).float()  # Convert probabilities to 0 or 1 based on threshold
        total += labels.size(0)
        correct += (predicted.squeeze(1) == labels).sum().item()
    return 100 * correct / total


def test_sex_resnet50(train_csv, val_csv, test_csv, img_directory):
    dataloader, dataloader_female, dataloader_male, _, model = resnet50(train_csv, val_csv, test_csv, img_directory)

    def get_predictions_and_probabilities(dataloader):
        # We get the predictions and probabilities of the model
        all_predictions = []
        all_probabilities = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                all_predictions.extend(predicted.squeeze().tolist())
                all_probabilities.extend(probabilities.squeeze().tolist())
        return all_predictions, all_probabilities

    # Then we get the predictions and probabilities for the dataloaders separately on males and females
    predicted, probabilities = get_predictions_and_probabilities(dataloader)
    predicted_female, probabilities_female = get_predictions_and_probabilities(dataloader_female)
    predicted_male, probabilities_male = get_predictions_and_probabilities(dataloader_male)

    return predicted, probabilities, predicted_female, probabilities_female, predicted_male, probabilities_male, model