# Face Mask Detection using YOLOv8

This project is an Object Detection system capable of detecting whether a person is wearing a face mask or not. It is built using the **YOLOv8** architecture (Ultralytics).

## Deployment
You can find and enjoy my deployed product at `[Deployment of Mask Wearing Detection with YOLOv8](https://huggingface.co/spaces/SaitoHoujou/Mask_Wearing_Detection)`

## Installation

1.  **Clone this repository** (or download the files).
2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
Then, you should check if your device has GPU for faster running.

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If your GPU exists but isn't enabled, you can use these scripts below to enable it in your virtual environment.

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 #This is cuda version suitable to my device, change if you need.
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```
## Dataset

Before running the training script, you need to place your dataset in the `dataset` folder:

1.  Put all your images in: `dataset/images/`
2.  Put all your XML annotation files in: `dataset/annotations/`

The `DataProcessor` module will automatically convert these into the correct YOLO format and create a `processed_data` folder during the first run.

## Run

To start the data processing and training pipeline, simply run the `main.py` file:

```bash
python main.py
```

**What happens when you run this?**

1.  **Data Processing:** Checks if data is prepared. If not, it converts XML to TXT, splits into Train/Val sets (80/20), and generates `data.yaml`.
2.  **Training:** Downloads `yolov8n.pt` and starts training for 50 epochs (default).
3.  **Validation:** Evaluates the model after training and prints mAP scores.

## Configuration

You can adjust hyperparameters and paths without touching the core code. Open `src/config.py` to change:

  * `epochs`: Number of training epochs (Default: 100).
  * `batch_size`: Batch size (Default: 16).
  * `device`: '0' for GPU or 'cpu'.
  * `raw_images_dir`: Path to your input images.

## Results

After training, the results (logs, confusion matrices, and model weights) will be saved in:
`./runs/detect/mask_detection_project/`

  * **Best Weights:** `./runs/detect/mask_detection_project/weights/best.pt`

## Credits

  * **YOLOv8** by [Ultralytics](https://github.com/ultralytics/ultralytics)
  * **Dataset:** [Face Mask Detection Dataset](https://drive.google.com/file/d/1nng7lMDXqTT4Hrmje0BCSf5nq-9jc3QF/view?usp=sharing )