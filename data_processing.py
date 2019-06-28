import numpy as np
from PIL import Image
from torchvision import transforms,datasets
import torch

def load_images(directory):
    '''
    Takes a directory and returns training, validation, and testing dataloaders.
    
    Note that the directory specified must have 3 sub-directores: "train", "valid", and "test".
    
    Each of these sub-directories must contain your images organized into sub-directories titled as their label.
        i.e. If training image "image123.jpg" has label 1, the "train" directory must have a "1" directory containing "image123.png"
    
    Input: Directory containing "train", "valid", and "test" directories
    Returns: Dataloaders: train_dataloader,valid_dataloader,test_dataloader, class_to_idx
    Note: Images are transformed to be cropped at 220x220 and normalized per PyTorch pretrained models. Dataloaders load in batches of 64.
    '''
    # DIRECTORIES
    data_dir = directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TRAINING DATA TRANSFORMATIONS
    data_transforms_train = transforms.Compose([transforms.Resize(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # TESTING DATA TRANSFORMATIONS
    data_transforms_test = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # IMPORTING IMAGES
    image_datasets_train = datasets.ImageFolder(train_dir,data_transforms_train)
    image_datasets_valid = datasets.ImageFolder(valid_dir,data_transforms_test)
    image_datasets_test = datasets.ImageFolder(test_dir,data_transforms_test)
    # CREATING DATALOADERS
    train_dataloader = torch.utils.data.DataLoader(image_datasets_train,batch_size=64,shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(image_datasets_valid,batch_size=64,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(image_datasets_test,batch_size=64,shuffle=True)

    try:
        class_to_idx = image_datasets_train.class_to_idx
    except:
        class_to_idx = None
        print("Warning: No class_to_idx mapping found.")

    return train_dataloader,valid_dataloader,test_dataloader,class_to_idx

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,

    Input: filepath to image

    Returns: NumPy Array of the image to be passed into the predict function
    '''
    # Open image
    im = Image.open(image).convert('RGB')
    # Resize keeping aspect ratio
    im.thumbnail(size=(256,256))
    # Get dimensions
    width, height = im.size
    # Set new dimensions for center crop
    new_width,new_height = 224,224 
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    im = im.crop((left, top, right, bottom))
    # Convert to tensor & normalize
    transf_tens = transforms.ToTensor()
    transf_norm = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    tensor = transf_norm(transf_tens(im))
    # Convert to numpy array
    np_im = np.array(tensor)
    return np_im

if __name__=='__main__':
    test_dir = "flowers"
    test_img = "flowers/test/99/image_07833.jpg"

    train_dataloader,valid_dataloader,test_dataloader = load_images('flowers')
    if train_dataloader and valid_dataloader and test_dataloader:
        print("Dataloaders successfully loaded")
    
    image = process_image(test_img)
    if image.any():
        print("Image successfully loaded.")