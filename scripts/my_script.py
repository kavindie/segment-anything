from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import cv2
from matplotlib import pyplot as plt

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

#device = "cuda"
sam = sam_model_registry["vit_h"](checkpoint="/scratch2/kat049/Git/segment-anything/sam_vit_h_4b8939.pth")
#sam.to(device=device)

predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

#image_path ='/scratch2/kat049/Git/STVT/STVT/STVT/datasets/TestDataset/Images/frame_10500.jpg'
image_path = '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/TrainDataset_3/Images/frame_98.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
#masks = mask_generator.generate(image)
#plt.figure(figsize=(20,20))
#plt.imshow(image)
#show_anns(masks)
#plt.axis('off')

input_label = np.array([1])
input_point = np.array([[750, 500]])

plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.savefig('/scratch2/kat049/Git/segment-anything/img.png')

masks, scores, logits = predictor.predict(
    "clothes"
)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)


 
plt.close()
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f'/scratch2/kat049/Git/segment-anything/img_{i}.png') 