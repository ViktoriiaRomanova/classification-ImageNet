import os
import random
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

__all__ = [
            'PreprocessingData',
            'ImageNetDataset',
            'get_not_RGB_pic',
            'conv_to_img'
]

DataType = Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]


class PreprocessingData:
    """Pair file names with class label and count class sizes."""

    def __init__(self) -> None:
        """Make variable to keep class sizes."""
        self.class_size: List[int] = []  # ind: class number, value: class size

    def get_data(self, data_path: str, random_state: int, numb: Optional[int] = None) -> DataType:
        """
            Collect data.

            Each class of photos was saved in it's own folder.
            Collecting all of them, assign class labels, splits into train and test/validation.
        """
        train_labels, test_labels = [], []  # list[tuple[image path, image lable], ...]
        for i, (_, _dirnames, filenames) in enumerate(os.walk(data_path)):
            if i == 0: continue  # first directory contains only folders, we don't need them

            # From each class we take 95% of data to train and 5% to test
            train, test = train_test_split(filenames, train_size = 0.95,
                                           shuffle = True, random_state = random_state)
            self.class_size.append((len(filenames)))

            train_labels.extend(zip(train, [i - 1] * len(train)))
            test_labels.extend(zip(test, [i - 1] * len(test)))
            # To get specified number of classes
            if numb is not None:
                numb -= 1
                if numb == 0: break
        random.seed(random_state)
        random.shuffle(test_labels)
        return train_labels, test_labels


class ImageNetDataset(Dataset):
    """Prepare data for DataLoader."""

    def __init__(self, img_dir: str, data: List[Tuple[str, int]],
                 transform: Union[nn.Module, transforms.Compose, None] = None) -> None:
        """
            Args:
                * dataset directory,
                * list of filenames with class indexes,
                * picture transformation.
        """
        self.imgsname_labels = data  # list[tuple[image path, image lable], ...]
        self.img_dir = img_dir
        self.transforms = transform

    def __len__(self) -> int:
        """Return number of pictures in dataset."""
        return len(self.imgsname_labels)

    def __getitem__(self, idx: int) -> Tuple[Union[Image.Image, torch.tensor], int]:
        """Return image/transformed image and it's class label by given index."""
        folder_name = self.imgsname_labels[idx][0].split('_')[0]
        img_path = os.path.join(self.img_dir, os.path.join(folder_name, self.imgsname_labels[idx][0]))

        image = Image.open(img_path)

        if self.transforms:
            image = self.transforms(image)
        return image, self.imgsname_labels[idx][1]


def get_not_RGB_pic(data: ImageNetDataset) -> Set[int]:
    """Get indexes of pictures which is not RGB."""
    indexes = set()
    for i in tqdm(range(len(data))):
        if not data[i][0].mode == 'RGB':
            indexes.add(i)
    return indexes


def conv_to_img(tensor: torch.tensor) -> np.array:
    """Convert image to display by pyplot."""
    img = tensor.to('cpu').clone().detach()
    img = img.numpy().squeeze()
    img = img.transpose(1, 2, 0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img.clip(0, 1)
    return img
