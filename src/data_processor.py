import os
import shutil
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
from src import config

class DataProcessor:
    def __init__(self):
        self.classes = config.classes
        self.working_dir = config.working_dir
        self._setup_directories()

    def _setup_directories(self):
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.working_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.working_dir, 'labels', split), exist_ok=True)

    def convert_box(self, size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        return (x*dw, y*dh, w*dw, h*dh)

    def convert_annotation(self, xml_file, output_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        with open(output_path, 'w') as out_file:
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in self.classes or int(difficult) == 1:
                    continue
                cls_id = self.classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = self.convert_box((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    def create_yaml(self):
        yaml_content = f"""
        train: {os.path.abspath(os.path.join(self.working_dir, 'images/train'))}
        val: {os.path.abspath(os.path.join(self.working_dir, 'images/val'))}

        nc: {len(self.classes)}
        names: {self.classes}
        """
        with open(config.data_yaml_path, 'w') as f:
            f.write(yaml_content)

    def process_data(self, split_ratio=0.8):
        all_xmls = [f for f in os.listdir(config.raw_annotations_dir) if f.endswith('.xml')]
        random.seed(42)
        random.shuffle(all_xmls)

        split_idx = int(len(all_xmls) * split_ratio)
        train_files = all_xmls[:split_idx]
        val_files = all_xmls[split_idx:]

        datasets = {'train': train_files, 'val': val_files}

        for split, files in datasets.items():
            print(f"Processing {split} data...")
            for xml_file in tqdm(files):
                image_name_png = xml_file.replace('.xml', '.png')
                image_name_jpg = xml_file.replace('.xml', '.jpg')
                
                src_img_path = None
                if os.path.exists(os.path.join(config.raw_images_dir, image_name_png)):
                    src_img_path = os.path.join(config.raw_images_dir, image_name_png)
                    dest_img_name = image_name_png
                elif os.path.exists(os.path.join(config.raw_images_dir, image_name_jpg)):
                    src_img_path = os.path.join(config.raw_images_dir, image_name_jpg)
                    dest_img_name = image_name_jpg
                
                if src_img_path:
                    shutil.copy(src_img_path, os.path.join(self.working_dir, 'images', split, dest_img_name))
                    self.convert_annotation(
                        os.path.join(config.raw_annotations_dir, xml_file),
                        os.path.join(self.working_dir, 'labels', split, xml_file.replace('.xml', '.txt'))
                    )
        
        self.create_yaml()