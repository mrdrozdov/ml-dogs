import os

import PIL
import scipy.io
import torch


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, metadata_path=None, images_folder=None, transform=None, target_transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.target_transform = target_transform

        filenames, labels = self.get_filenames_and_labels(metadata_path)

        self.filenames = filenames
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_name = self.filenames[index]
        image_path = os.path.join(self.images_folder, image_name)
        image = PIL.Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_filenames_and_labels(self, metadata_path):
        annotations = scipy.io.loadmat(metadata_path)
        filenames = [x[0][0] + '.jpg' for x in annotations['annotation_list']]
        labels = [x[0] - 1 for x in annotations['labels']]

        assert min(labels) == 0
        assert len(set(labels)) == max(labels) + 1

        return filenames, labels
