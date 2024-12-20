import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from torchvision.utils import save_image
import os
from PIL import Image

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


root_dir='Data folder/val/malignant'
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
for _ in range(3):
    for img in combined_dataset:
        save_path = os.path.join(root_dir, f"augmented_img{img_num}.png")
        save_image(img, save_path)
        img_num+=1

train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=4)

new_dataset = CustomDataset(root_dir=root_dir, transform=None)
new_train_B_num = len(new_dataset)
print(f"Number of images after transformation: {new_train_B_num}")
