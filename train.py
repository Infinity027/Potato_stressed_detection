import os
import argparse
import torch
from tqdm.auto import tqdm
import time
from pathlib import Path
# import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchmetrics.detection import MeanAveragePrecision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader.augmentation import RandomHorizontalFlip, RandomVerticalFlip
from dataloader.dataset import SpectralDataset
from model.custom_model import CustomYolo
from utils.loss import YoloLoss
from utils.plot import plot_loss, plot_image
from utils.evalute import (decode_predictions, process_targets, 
                           accuracy, load_checkpoint, save_checkpoint, 
                           print_gpu_memory, get_data_memory, get_model_memory)

CLASS_NAME= {
  0: 'healthy',
  1: 'stressed'
}
ROOT = Path(__file__).resolve().parents[0]
SEED = 2023
torch.manual_seed(SEED)

def arg_parser():
    parser = argparse.ArgumentParser(description="YOLOv1 Implementation")
    parser.add_argument("--data_dir", type=str, default="potato/labels", help="Path label data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--nms_thres", type=float, default=0.75, help="NMS Threshold")
    parser.add_argument("--iou_thres", type=float, default=0.75, help="IOU Threshold")
    parser.add_argument("--resume", type=bool, default=False, help="load last checkpoint")

    args = parser.parse_args()
    if os.path.exists(args.data_dir) == False:
        raise FileNotFoundError("Data file not found")
    return args

def train_step(model, loader, criterion, optimizer, device):
    total_loss = 0
    box_loss = 0
    object_loss = 0
    no_object_loss = 0
    class_loss = 0
    acc = 0

    model.train()
    for i, (images, targets) in enumerate(loader):
        print(f"Batch:{i}/{len(loader)}",end=' ')
        # start = time.time()
        images = images.to(device)
        # print_gpu_memory()
        preds = model(images)
        # torch.cuda.empty_cache()
        # print_gpu_memory()
        loss, obj_loss, noobj_loss, b_loss, cls_loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # end = time.time()
        # print("Execute time:",(end-start))
        total_loss += loss.item()
        object_loss += obj_loss.item()
        no_object_loss += noobj_loss.item()
        box_loss += b_loss.item()
        class_loss += cls_loss.item()
        acc = accuracy(preds, targets)

        print(f"Loss:{loss:.4f} | Box Loss:{b_loss:.4f} | Object Loss:{obj_loss:.4f} | No Object Loss:{noobj_loss:.4f} | Class Loss:{cls_loss:.4f}",end='\r')
    # del images, preds
    # torch.cuda.empty_cache()

    total_loss /= len(loader)
    box_loss /= len(loader)
    object_loss /= len(loader)
    no_object_loss /=len(loader)
    class_loss /= len(loader)
    acc /= len(loader)

    return {
        "total_loss": total_loss,
        "box_loss": box_loss,
        "object_loss": object_loss,
        "no_object_loss": no_object_loss,
        "class_loss": class_loss,
        "acc": acc
    }

def val_step(model, loader, criterion, device, conf_thresh=0.5, epoch=0):
    total_loss = 0
    box_loss = 0
    object_loss = 0
    class_loss = 0
    no_object_loss = 0
    acc = 0

    # Initialize mAP metric
    metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox")

    model.eval()
    if True:
        draw = torch.randint(0,len(loader),size=(1,)).item()
    else:
        draw = -1
    with torch.inference_mode():
        #target is a list of present object [class, xc, yc, w, h]
        for i, (images, targets) in enumerate(loader):
            print(f"Batch:{i}/{len(loader)}",end=' ')
            images = images.to(device)
            preds = model(images)
            loss, obj_loss, noobj_loss, b_loss, cls_loss = criterion(preds, targets)

            # mAP prcoess
            if i==draw:
                pred_boxes, idxs = decode_predictions(images, preds, C=2, S=7, conf_thresh=conf_thresh, plot=epoch)
                select_target = [targets[idx] for idx in idxs]
                save_name = f"results/true_plot_{epoch}.png"

                plot_image(images[idxs].to("cpu"), select_target, CLASS_NAME, save_name=save_name)
            else:
                pred_boxes = decode_predictions(images, preds, C=2, S=7, conf_thresh=conf_thresh)

            target_boxes = process_targets(targets)
            metric.update(pred_boxes, target_boxes)

            total_loss += loss.item()
            object_loss += obj_loss.item()
            no_object_loss += noobj_loss.item()
            box_loss += b_loss.item()
            class_loss += cls_loss.item()
            acc += accuracy(preds,targets)
            print(f"Loss:{loss:.4f} | Box Loss:{b_loss:.4f} | Object Loss:{obj_loss:.4f} | No Object Loss:{noobj_loss:.4f} | Class Loss:{cls_loss:.4f}",end='\r')

    # Compute mAP
    map_metrics = metric.compute()

    total_loss /= len(loader)
    box_loss /= len(loader)
    object_loss /= len(loader)
    no_object_loss /=len(loader)
    class_loss /= len(loader)
    acc /= len(loader)

    return {
        "total_loss": total_loss,
        "box_loss": box_loss,
        "object_loss": object_loss,
        "no_object_loss": no_object_loss,
        "class_loss": class_loss,
        "acc": acc,
        "map": map_metrics["map"],          # Mean Average Precision
        "map_50": map_metrics["map_50"],    # mAP@IoU=0.5
        "map_75": map_metrics["map_75"],    # mAP@IoU=0.75
    }

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=100):
    train_result = {
        'total_loss':[],
        'box_loss':[],
        'object_loss':[],
        'no_object_loss':[],
        'class_loss':[],
        'acc': []
    }
    val_result = {
        'total_loss':[],
        'box_loss':[],
        'object_loss':[],
        'no_object_loss':[],
        'class_loss':[],
        'acc': [],
        'map':[],
        'map_50':[],
        'map_75':[],
    }

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {(epoch+1)}/{epochs}:")
        result = train_step(model, train_loader, criterion, optimizer, device)
        for key in result.keys():
            train_result[key].append(result[key])

        print("Training: ",end=" ")
        print(f"Total Loss:{result['total_loss']:.4f} | Box Loss:{result['box_loss']:.4f} | Object Loss:{result['object_loss']:.4f} | No Object Loss:{result['no_object_loss']:.4f} | Class Loss:{result['class_loss']:.4f}")

        result = val_step(model, val_loader, criterion, device, epoch=epoch)
        for key in result.keys():
            val_result[key].append(result[key])

        print("Validation: ",end=" ")
        print(f"Total Loss:{result['total_loss']:.4f} | Box Loss:{result['box_loss']:.4f} | Object Loss:{result['object_loss']:.4f} | No Object Loss:{result['no_object_loss']:.4f} | Class Loss:{result['class_loss']:.4f}")
        print(f"Accuarcy:{result['acc']:.4f} | mAP:{result['map']:.4f} | mAP_50:{result['map_50']:.4f} | mAp_75:{result['map_75']:.4f}")
        scheduler.step(result['total_loss'])
        if (epoch+1)%10==0:
            print("model saved")
            checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

    return train_result, val_result

if __name__ == "__main__":
    args = arg_parser()
    torch.cuda.empty_cache() 
    device = "cpu"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    #load train and test data
    train_path = ROOT / args.data_dir / "train"
    test_path = ROOT / args.data_dir / "test"

    train_dataset = SpectralDataset(train_path, transform=trans["train"])
    val_dataset = SpectralDataset(test_path, transform=trans["val"])
    print("Total training data:", train_dataset.__len__())
    print("Total Testing data:", val_dataset.__len__())

    # create data loader
    train_loader = DataLoader(train_dataset, 
                              collate_fn=SpectralDataset.collate_fn, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              pin_memory=True,
                              num_workers = 2)
    
    val_loader = DataLoader(val_dataset, 
                            collate_fn=SpectralDataset.collate_fn, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            pin_memory=True)
    
    model = CustomYolo().to(device)
    print("--------model load complete--------")
    # print_gpu_memory()

    loss_fn = YoloLoss()
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5, threshold=0.01)

    if args.resume:
        load_checkpoint(torch.load("weights/my_checkpoint.pth"), model, opt)

    train_r, val_r= train(model, train_loader, val_loader, loss_fn, opt, scheduler, device, epochs=args.epochs)

    plot_loss(train_r, save_name="results/train_loss.png")
    plot_loss(val_r, save_name="results/val_loss.png")

    torch.save(obj=model,f='weights/last.pt')

