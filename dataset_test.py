import cv2
from torch.utils.data import Dataset
import os
from PIL import Image


class Mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset"
apple_label_dir = "apple"
banana_label_dir = "banana"
apple_dataset = Mydata(root_dir, apple_label_dir)
banana_dataset = Mydata(root_dir, banana_label_dir)
train_dataset = banana_dataset+apple_dataset