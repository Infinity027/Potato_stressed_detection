from torchvision.transforms import functional as F, v2, transforms as T
import torch 
import os
from PIL import Image
import sys
from pathlib import Path
import random as rd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.plot import plot_image_with_label

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            #target conatins -- [cls, x_center, y_center, w, h]
            if target is not None:
                target[:, 1] = 1 - target[:, 1]
        return img, target
    
class RandomVerticalFlip(T.RandomVerticalFlip):
    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            img = F.vflip(img)
            #target conatins -- [cls, x_center, y_center, w, h]
            if target is not None:
                target[:, 2] = 1 - target[:, 2]
        return img, target

if __name__ == "__main__":
    CLASS_INFO= {
        0: 'stressed',
        1: 'healthy'
}
    
    trans = v2.Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(brightness=0.5, contrast=0.5)], p=0.5),
        v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
        v2.RandomPosterize(bits=4, p=0.2),
        v2.RandomSolarize(threshold=128, p=0.2)
    ])

    images, img_labels = [], []
    imgs_path = "data/images/val"
    k = 4
    filenames = os.listdir(imgs_path)
    #choose k random images and labels
    for filename in rd.choices(filenames, k=k):
        img_path = os.path.join(imgs_path, filename)
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = v2.ToImage()(img)
        label_path = img_path.replace("images", "labels").replace("jpg", "txt")
        with open(label_path, mode="r") as f:
            labels = [x.split() for x in f.read().splitlines()]
        f.close()

        #labels convert to tensor
        labels = torch.tensor([[float(x) for x in label] for label in labels])
        img, labels = trans(img, labels)
        images.append(img)
        img_labels.append(labels)

    for label in img_labels:
        print(label)
    plot_image_with_label(images, img_labels, CLASS_INFO)


