import torch
import cv2
import os
from pathlib import Path
import sys
import numpy as np
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataloader.augmentation import RandomHorizontalFlip, RandomVerticalFlip

class SpectralDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, image_size=448, transform = None):
        '''
        
        '''
        # load labels path
        self.label_paths = [os.path.join(dir_path,fn) for fn in os.listdir(dir_path)]
        
        #load labels path
        self.label2image_path()
        self.class_dict = {
            0: "healthy",
            1: "stressed"
        }

        self.size = image_size
        self.transform = transform

    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, index):
        """Fetches the image and label by index"""
        im = torch.zeros(4,self.size,self.size)

        for idx, image_path in enumerate(self.image_paths[index]):
            # Resize and assign to the channel
            # print(image_path)
            im[idx] = torch.from_numpy(cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), 
                                                  (self.size, self.size)))/255
            
        # load lables
        with open(self.label_paths[index], mode="r") as f:
            labels = [x.split() for x in f.read().splitlines()]
        f.close()

        labels = torch.tensor([[float(x) for x in label] for label in labels])
        if self.transform:
           im, labels = self.transform(im, labels)

        return im, labels

    def label2image_path(self):
        '''
        replace the label path to image path and also change extension
        '''
        l = f"{os.sep}labels{os.sep}" 
        red_l = f"{os.sep}Red_Channel{os.sep}"
        rededge_l = f"{os.sep}Red_Edge_Channel{os.sep}"
        green_l = f"{os.sep}Green_Channel{os.sep}"
        infared_l = f"{os.sep}Near_Infrared_Channel{os.sep}"
        
        self.image_paths = []
        for path in self.label_paths:
            red_path = red_l.join(path.rsplit(l, 1)).rsplit(".", 1)[0] + ".jpg"
            rededge_path = rededge_l.join(path.rsplit(l, 1)).rsplit(".", 1)[0] + ".jpg"
            green_path = green_l.join(path.rsplit(l, 1)).rsplit(".", 1)[0] + ".jpg"
            infared_path = infared_l.join(path.rsplit(l, 1)).rsplit(".", 1)[0] + ".jpg"
            self.image_paths.append([red_path, rededge_path, green_path, infared_path])
    
    @staticmethod
    def collate_fn(minibatch):
        images, labels = zip(*minibatch)
        images = torch.stack(images, dim=0)
        # labels is a tuple of variable-length tensors; leave them as is or process accordingly.
        return images, labels

if __name__ == "__main__":
    from utils.plot import plot_image_with_label
    train_path = ROOT / "potato" / "labels" / "train"
    val_path = ROOT / "potato" / "labels" / "test"

    # define augmentation
    trans = {
        "train": v2.Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
            v2.ToDtype(torch.float32, scale=True)
        ]), 
        "val": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])
    }

    traindata = SpectralDataset(dir_path = train_path, transform=None)
    print("Total data:",traindata.__len__())
    valdata = SpectralDataset(dir_path = val_path, transform=None)
    print("Total data:",valdata.__len__())
    # print(traindata.__getitem__(1))
    k = 4
    # choose random k images
    idxs = np.random.choice(range(traindata.__len__()), k)

    images = []
    labels = []
    rgb = []
    for idx in idxs:
        image, label = traindata.__getitem__(idx)
        print(traindata.label_paths[idx])
        rgb_img = cv2.imread(traindata.label_paths[idx].replace("labels", "rgb").replace("txt", "jpg"))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb.append(cv2.resize(rgb_img, (448,448))) 
        images.append(image)
        labels.append(label)

    plot_image_with_label(images, rgb, labels, traindata.class_dict, channel=0, save_name="red_channel.png")
    plot_image_with_label(images, rgb, labels, traindata.class_dict, channel=1, save_name="rededge_channel.png")
    # plot_image_with_label(images, rgb, labels, traindata.class_dict, channel=2, save_name="green_channel.png")
    # plot_image_with_label(images, rgb, labels, traindata.class_dict, channel=3, save_name="infared_channel.png")

