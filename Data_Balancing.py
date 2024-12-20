import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from torchvision.utils import save_image
import os
from PIL import Image
import random

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image


root_dir='Data folder/train/malignant'
original_dataset = CustomDataset(root_dir=root_dir, transform=None)
orgi_train_B_num = len(original_dataset)
print(f"Number of images before transformation: {orgi_train_B_num}")


augmentations_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
   # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmentations_2 = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmentations_3 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(50),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


lengths = [len(original_dataset) // 3, len(original_dataset) // 3, len(original_dataset) - 2 * (len(original_dataset) // 3)]
subset1, subset2, subset3 = random_split(original_dataset, lengths)


subset1.dataset.transform = augmentations_1
subset2.dataset.transform = augmentations_2
subset3.dataset.transform = augmentations_3

combined_dataset = ConcatDataset([subset1, subset2, subset3])

img_num=0
for _ in range(2):
    for img in combined_dataset:
        save_path = os.path.join(root_dir, f"augmented_img{img_num}.png")
        save_image(img, save_path)
        img_num+=1

train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=4)

new_dataset = CustomDataset(root_dir=root_dir, transform=None)
new_train_B_num = len(new_dataset)
print(f"Number of images after transformation: {new_train_B_num}")


def reduce_images_in_folder(folder_path, reduction_ratio):
    # List all files in the folder
    all_files = os.listdir(folder_path)
    
    # Filter to include only image files (common image extensions)
    image_files = [f for f in all_files if f.lower().endswith('.jpg')]
    
    print(f'Number of images before undersampling: {len(image_files)}')

    # Calculate the number of files to delete
    num_files_to_delete = int(len(image_files) * reduction_ratio)
    
    # Randomly select files to delete
    files_to_delete = random.sample(image_files, num_files_to_delete)
    
    # Delete the selected files
    for file in files_to_delete:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)

    remaining_image_files = len(image_files) - num_files_to_delete
    print(f'Number of images after undersampling: {remaining_image_files}')

    
# Example usage
folder_path = 'Data folder/train/benign'
reduction_ratio = 0.27 

reduce_images_in_folder(folder_path, reduction_ratio)
