import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("./runs/detect/mask_detection_model/weights/best.pt")

def predict_image(image):
    if image is None:
        return None

    results = model.predict(image, conf=0.5)
    
    res_plotted = results[0].plot()
    
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    return res_rgb

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Uploaded Image"),
    outputs=gr.Image(type="numpy", label="Detecting Result"),
    title="Mask Wearing Detection System with YOLOv8",
)

if __name__ == "__main__":
    iface.launch(share=False)