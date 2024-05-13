import pandas as pd
from PIL import Image
import os
import keyboard

df = pd.read_csv('data/metadata/master_data.csv')

# Filter DataFrame to select images with gender='FEMALE'
female_images = df[df['gender'] == 'FEMALE']

# Path to the directory containing the images
image_dir = 'data/images/lesion_images/'

# Function to display image and img_id
def display_image(img_id):
    image_path = os.path.join(image_dir, img_id)
    img = Image.open(image_path)
    img.show()
    print("Image ID:", img_id)
    return img

def close_image(img):
    img.close()

def display_batch(batch_number):
    start_index = batch_number * 100
    end_index = min((batch_number + 1) * 100, len(female_images))
    images = []
    for i in range(start_index, end_index):
        img_id = female_images.loc[i, 'img_id']
        img = display_image(img_id)
        images.append(img)

        input("Press Enter to view the next image...")  # Wait for user input to continue
        keyboard.wait('enter')
        if images:
            images.pop().close()


batch_number = int(input("Enter batch number (0, 1, or 2): "))

# Display the corresponding batch of 100 images
if batch_number in [0, 1, 2]:
    display_batch(batch_number)
else:
    print("Invalid batch number. Please enter 0, 1, or 2.")
