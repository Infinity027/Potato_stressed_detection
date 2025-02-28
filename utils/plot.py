import numpy as np
import matplotlib.pyplot as plt
# from PIL import ImageDraw, ImageFont
from matplotlib.patches import Rectangle

def plot_image(images, labels, class_names, save_name="results/plot_image.png"):
    """
    draw image and label using plot function
    """
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'white']
    class_names[-1] = "None"
    #only first 25 images draw
    if len(images)>16:
        images = images[:16]
        labels = labels[:16]

    row = int(np.ceil(np.sqrt(len(images))))
    col = row if row * (row - 1) < len(images) else row + 1
    fig, axes = plt.subplots(row, col, figsize=(col * 5, row * 5))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    im_size = images[0].shape[1]
    for image, boxes, ax in zip(images, labels, axes):
        if image.shape[0] == 3:
            ax.imshow(image.permute(1,2,0))
        else:
            ax.imshow(image)
        for box in boxes:
            # print(box)
            if len(box)>5:
                cls_id, conf, x, y, w, h = box
                names = f"{class_names[int(cls_id)]}: {(conf*100).item():.2f}"
            else:
                cls_id, x, y, w, h = box
                names = f"{class_names[int(cls_id)]}"

            x, y, w, h = x*im_size, y*im_size, w*im_size, h*im_size
            rect = Rectangle((x-w/2, y-h/2), w, h, edgecolor=colors[int(cls_id)], facecolor='none')
            ax.add_patch(rect)
            ax.text(x-w/2, y-h/2, names, fontsize=9, color=colors[int(cls_id)])
            ax.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_name)
    plt.close()

def plot_image_with_label(images, rgb, labels, class_names, channel = 0, save_name="results/plot_image_with_label.png"):
    """
    draw image and label using plot function
    """
    channel_name = {0:"Red", 
                    1:"Red Edge", 
                    2:"Green", 
                    3:"Infared"
                    }
    
    colors = ['red', 'blue', 'white']
    class_names[-1] = "None"
    #only first 5 images draw
    if len(images)>5:
        images = images[:5]
        labels = labels[:5]
    col = len(images)

    fig, axes = plt.subplots(2, col, figsize=(col * 5, 2 * 5))

    im_size = images[0].shape[1]
    for i in range(len(images)):
        axes[0,i].imshow(images[i][1:].permute(1,2,0)/255)
        axes[0,i].set_title(f"Channel: {channel_name[channel]}")

        axes[1,i].imshow(rgb[i])
        axes[1,i].set_title(f"Channel: RGB")
        for box in labels[i]:
            
            cls_id, x, y, w, h = box
            names = f"{class_names[int(cls_id)]}"

            x, y, w, h = x*im_size, y*im_size, w*im_size, h*im_size
            rect = Rectangle((x-w/2, y-h/2), w, h, edgecolor=colors[int(cls_id)], facecolor='none')
            axes[0,i].add_patch(rect)
            axes[0,i].text(x-w/2, y-h/2, names, fontsize=9, color=colors[int(cls_id)])
            axes[0,i].axis("off")
            rect = Rectangle((x-w/2, y-h/2), w, h, edgecolor=colors[int(cls_id)], facecolor='none')
            axes[1,i].add_patch(rect)
            axes[1,i].text(x-w/2, y-h/2, names, fontsize=9, color=colors[int(cls_id)])
            axes[1,i].axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_name)
    plt.close()

def plot_loss(result, save_name="results/loss.png"):
    """
    draw losses and accuarcy
    """
    if len(result.keys())>6:
        plt.figure(figsize=(14,12))
        row = 3
        col = 3
    else:
        plt.figure(figsize=(14,8))
        row = 2
        col = 3 

    for i, key in enumerate(result.keys()):
        plt.subplot(row,col,i+1)
        plt.plot(result[key])
        plt.title(key)
        plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

if __name__=="__main__":
    train_loss = {
        'total_loss':[],
        'box_loss':[],
        'object_loss':[],
        'no_object_loss':[],
        'class_loss':[],
        'acc':[]
    }
    for i in range(100):
        train_loss['total_loss'].append(np.random.randint(100))
        train_loss['box_loss'].append(np.random.randint(100))
        train_loss['object_loss'].append(np.random.randint(100))
        train_loss['no_object_loss'].append(np.random.randint(100))
        train_loss['class_loss'].append(np.random.randint(100))
        train_loss['acc'].append(np.random.randint(100))

    plot_loss(result=train_loss)