import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
import tensorflow as tf


class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        img = self.BGR2RGB(img)  # because "torchvision.utils.save_image" use RGB
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    transform = transforms.Compose(
        [transforms.ToPILImage(),  # convert a tensor to PIL image
         transforms.Resize((64, 64)),  # resize the picture to given size
         transforms.ToTensor(),  # convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
         transforms.Normalize(mean=(0.5, 0.5, 0.5),  #([0.0,1.0] - 0.5)/0.5 ->[-1,1]
                              std=(0.5, 0.5, 0.5))])  # Normalized an tensor image with mean and standard deviation
    dataset = FaceDataset(fnames, transform)
    return dataset
