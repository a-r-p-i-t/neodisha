import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import requests
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor




sam_checkpoint = "C:\\Users\\Arpit Mohanty\\Desktop\\neo_disha\\models\\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device="cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)



def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))



def normalize_points(contour_points, image_width, image_height):
    # Normalize the points
    normalized_points = contour_points.astype(float) / np.array([image_width, image_height])
    return normalized_points




def convert_contour_to_yolov8(contour_points, class_index):
    # Flatten the contour points to a 1D array
    flattened_points = contour_points.reshape(-1, 2)

    # Create YOLOv8 label string
    label_string = f"{class_index}"
    for point in flattened_points:
        label_string += f" {point[0]} {point[1]}"
    return label_string



def download_image_from_drive(url, destination):
    response = requests.get(url)
    response.raise_for_status()
    with open(destination, 'wb') as file:
        file.write(response.content)
    print("Image downloaded successfully.")



image_url = ""
destination_path = "C:\\Users\\Arpit Mohanty\\Desktop\\neo_disha\\data\\images\\image.jpg"




def extract_roi(image_path,coord_list):
    image=cv2.imread(image_path)
    x1, y1, x2, y2 = coord_list
    roi = image[y2:y1, x1:x2]
    return roi


def load_model(model_path):
    model=YOLO(model=model_path)
    return model



def extract_labels(roi,model_path):

    model=load_model(model_path)
    img_dir="C:\\Users\\Arpit Mohanty\\Desktop\\neo_disha\\data\\images"
    img_list = os.listdir(img_dir)
    img_list.sort()
    image_path=img_dir + img_list[0]

    result=model.predict(source=roi,conf=0.5,save=True,show_labels=False)
    contours=[]
    boxes=result.boxes

    i = 0
    label_string_list =[]
    while i < len(boxes.xyxy.tolist()):
        bbox = boxes.xyxy.tolist()[i]
        class_id = boxes.cls.tolist()[i]
        # print(bbox)

        image = cv2.cvtColor(cv2.imread(roi), cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        input_box = np.array(bbox)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        # print(masks)
        mask = masks[0]
        # print(mask)
        contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(contour)

        contour_points = contours[i]
        class_index = int(class_id)
        image_height, image_width = image.shape[:2]

        normalized_points = normalize_points(contour_points, image_width, image_height)
        label_string = convert_contour_to_yolov8(normalized_points, class_index)
        label_string_list.append(label_string)
        i += 1
    output_path = 'C:\\Users\\Arpit Mohanty\\Desktop\\neo_disha\\data' + image_path.split(".")[0] + '.txt'
    print(output_path)
    with open(output_path, 'w') as file:
        for item in label_string_list:
            file.write(str(item) + '\n')

bounding_box_list=[]
model_path=""
download_image_from_drive(image_url, destination_path)
roi=extract_roi(destination_path,bounding_box_list[0])
extract_labels(roi,model_path)








    