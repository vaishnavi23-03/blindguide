import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import pyttsx3

def text_to_audio(text):
    # Initialize the pyttsx3 engine
    engine = pyttsx3.init()
    
    # Set properties (optional, adjust voice rate, volume, etc.)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)
    
    # Speak the text
    engine.say(text)
    engine.runAndWait()

# Load models once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

yolo_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
yolo_model.eval()

seg_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
seg_model.eval()

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
# Use same COCO class list as in your code

scene_text = ["a room", "a street", "a park", "an office", "a chemistry lab"]

def process_frame(frame):
    # Convert to PIL and resize
    print("Processing frame")
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize((512, 512))

    # SCENE CLASSIFICATION
    print("Scene classification")
    inputs = clip_processor(images=pil_img, text=scene_text, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    scene_pred = scene_text[torch.argmax(probs, dim=1).item()]
    
    # Output the scene classification
    print(f"Predicted scene: {scene_pred}")
    text_to_audio(f"The scene is {scene_pred}")

    # SEMANTIC SEGMENTATION
    print("Semantic segmentation")
    transform_seg = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_seg = transform_seg(pil_img).unsqueeze(0)
    with torch.no_grad():
        seg_output = seg_model(input_seg)['out'][0].cpu().numpy()

    # Check if segmentation output is valid
    if seg_output is not None and seg_output.shape[0] > 0:
        # Segmentation mask (assuming output is a single-channel segmentation map)
        seg_mask = seg_output[0]  # Get the first channel (segmentation mask)

        # Ensure mask is valid and resize it to match the input image size
        seg_mask_resized = cv2.resize(seg_mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Overlay segmentation
        seg_overlay = (seg_mask_resized > 0.5).astype(np.uint8) * 255
        seg_overlay = cv2.merge([seg_overlay] * 3)
        overlay = cv2.addWeighted(img_rgb, 0.8, seg_overlay, 0.2, 0)
    else:
        # In case the segmentation output is invalid, just return the original image
        overlay = img_rgb

    # OBJECT DETECTION
    print("Object detection")
    img_tensor = transforms.ToTensor()(Image.fromarray(img_rgb)).unsqueeze(0)
    with torch.no_grad():
        prediction = yolo_model(img_tensor)

    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            class_name = COCO_CLASSES[label]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(overlay, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return overlay, scene_pred
