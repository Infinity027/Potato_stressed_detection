import torch
import torchvision.ops as ops
import numpy as np


def decode_predictions(images, predictions, C, S=7, conf_thresh=0.5, nms_thresh=0.6, plot=None):
    """
    Convert model output to list of bounding boxes in cxcywh format.
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    pred_boxes = []
    
    for i in range(batch_size):
        #pred shape: [S*S, (C + 5*B)]
        pred = predictions[i]
        
        # Get class probabilities and bounding boxes
        class_probs = pred[:, :C]
        box_data = pred[:, C:]

        #converting to absolute center point
        for i, box in enumerate(box_data):
            box[:,1] = (box[:,1] + i%S)/S
            box[:,2] = (box[:,2] + i//S)/S

        mask = box_data[:,0]>conf_thresh

        boxes = box_data[mask,1:]
        scores = box_data[mask,0]
        labels = torch.argmax(class_probs[mask], dim=-1)

        #apply nms 
        if boxes.shape[0] > 0:
            keep_indices = ops.nms(boxes, scores, nms_thresh)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
        else:
            boxes = torch.empty((0, 4))
            scores = torch.empty((0,))
            labels = torch.empty((0,), dtype=torch.int64)

        pred_boxes.append({
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        })

    select_pred = []
    if plot!=None:
        from utils.plot import plot_image_with_label
        idxs = np.random.choice(range(batch_size),4)
        for idx in idxs:
            select_pred.append(torch.cat((pred_boxes[idx]['labels'].unsqueeze(-1), 
                                         pred_boxes[idx]['scores'].unsqueeze(-1), 
                                         pred_boxes[idx]['boxes']), -1))
            
        class_name = {0:'healthy',
                      1:'stressed'
                      }
        
        save_name = f"results/pred_plot_{plot}.png"
        plot_image_with_label(images[idxs].to("cpu"), select_pred, class_name, save_name)

        print(select_pred)

        return pred_boxes, idxs
    else:
        return pred_boxes

def process_targets(targets):
    """
    Convert targets to list of dictionaries with "boxes" and "labels".
    """
    true_boxes = []
    for target in targets:
        boxes = []
        labels = []
        for obj in target:
            #target format: [class, cx, cy, w, h]
            class_id = int(obj[0])
            cx, cy, w, h = obj[1], obj[2], obj[3], obj[4]
            
            boxes.append([cx, cy, w, h])
            labels.append(class_id)
        
        true_boxes.append({
            "boxes": torch.tensor(boxes),
            "labels": torch.tensor(labels),
        })
    return true_boxes

def accuracy(pred, target, S=7, C=5):
    """
    Calculate classification accuracy for objects in grid cells.
    
    Args:
        pred (Tensor): Model output of shape [batch_size, S*S, (C + 5*B)].
        target (list): List of tensors, each containing objects [cls_id, xc, yc, w, h].
        S (int): Grid size (default=7).
        C (int): Number of classes (default=5).
    
    Returns:
        float: Classification accuracy.
    """
    class_pred = []
    pred_cls = pred[..., :C].view(-1, S, S, C)  # Reshape to [batch, S, S, C]
    
    for batch_pred, batch_target in zip(pred_cls, target):
        for obj in batch_target:
            
            # Check if prediction matches target
            class_pred.append(torch.argmax(batch_pred[min(int(obj[1].item()*S),S-1), 
                                                      min(int(obj[2].item()*S),S-1)]).item() == int(obj[0].item()))
    
    # Calculate accuracy
    correct = sum(class_pred)
    total = len(class_pred)

    return correct / total if total > 0 else 0.0

def save_checkpoint(state, filename="weights/my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    cached = torch.cuda.memory_reserved() / 1024**2      # MB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"Allocated: {allocated:.2f} MB | Cached: {cached:.2f} MB | Total: {total:.2f} MB")

def get_data_memory(batch, dtype=torch.float32):
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    data_mem = batch.element_size() * batch.numel() / (1024**2)  # MB
    return data_mem

def get_model_memory(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / (1024**2)  # Convert to MB
    return total_size

if __name__=="__main__":

    pred = torch.rand(16,25,15)
    images = torch.ones(16,32,32,3)*0.5
    # print(pred)
    pred_boxes, select_pred = decode_predictions(images, pred, C=5, S=5, plot=True)
    print(select_pred)
    # for pred in pred_boxes:
    #     for key in pred.keys():
    #         print(f"{key}:{pred[key]}")     
    # target = [[torch.Tensor([0, .2, .5, .01, .02]), torch.Tensor([1, .3, .8, .01, .02]), torch.Tensor([2, .4, .1, .01, .02]),],
    #             [torch.Tensor([1, .4, .4, .01, .02]), torch.Tensor([2, .1, .1, .01, .02]), torch.Tensor([3, .2, .5, .01, .02]), torch.Tensor([3, .3, .3, .03, .02])],
    #             [torch.Tensor([3, .1, .1, .01, .02]), torch.Tensor([4, .1, .9, .01, .02]), torch.Tensor([4, .9, .3, .01, .02])],
    #             [torch.Tensor([4, .7, .5, .1, .02]), torch.Tensor([0, .3, .7, .01, .02])]]
    
    # print(accuracy(pred, target, S=2, C=5))