import os
import random

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
folder_path = 'Data folder/val/benign'
reduction_ratio = 0.7  # Reduce the number of images by 50%

reduce_images_in_folder(folder_path, reduction_ratio)
