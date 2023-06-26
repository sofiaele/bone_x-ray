
# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from PIL import Image, ImageFile
import os
from numpy import asarray
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomVisionDataset(Dataset):
    def __init__(self, dataframe, mean, std, mode=None, rgb=True):
        self.X = dataframe['path'].values.tolist()
        self.y = dataframe['label'].values.tolist()
        self.rgb = rgb

        # -------------------------------------------------------------------------------
        # image transformations
        if mode == 'test':
            self._transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                #using mean and std calculated on train data
                transforms.Normalize(mean, std)])
        else:
            self._transforms = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.Resize((224, 224)),
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                #transforms.Resize(256),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        image_path = self.X[index]
        image = Image.open(os.path.join(image_path))
        if self.rgb:
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        pic = asarray(image)
        # convert from integers to floats
        pic = pic.astype('float32')
        # normalize to the range 0-1
        pic /= 255.0
        image = Image.fromarray(pic.astype('uint8'))
        image = self._transforms(image)
        return image, self.y[index], image_path
