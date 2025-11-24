import os

raw_images_dir = './dataset/images'
raw_annotations_dir = './dataset/annotations'

working_dir = './processed_data' 
data_yaml_path = os.path.join(working_dir, 'data.yaml')

model_name = 'yolov8n.pt'
project_name = 'mask_detection'
classes = ['with_mask', 'without_mask']

epochs = 100
batch_size = 16
image_size = 640
device = 0       
workers = 0       