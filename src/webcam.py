import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
from models.convnextiny import ConvNeXtTinyEmotion

CONFIG = {
    "model_path": "../checkpoints/convnext_hyper_optimized_best.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": 224,
    "padding": 0.25,
    "smooth_factor": 0.7,   
}

CLASS_NAMES = ['Surprised', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
COLORS = {
    'Happy': (0, 255, 0), 'Sad': (255, 0, 0), 'Angry': (0, 0, 255),
    'Surprised': (0, 255, 255), 'Fear': (255, 0, 255), 
    'Disgust': (0, 128, 255), 'Neutral': (200, 200, 200)
}

def load_trained_model():
    print("Chargement du modèle avec la classe personnalisée...")
    
    model = ConvNeXtTinyEmotion(num_classes=len(CLASS_NAMES), pretrained=False, dropout_p=0.4)
    
    try:

        checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'], weights_only=True)

        state_dict = checkpoint.get("model_state", checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k == "n_averaged": continue
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=True)
        print("Modèle chargé avec succès !")
        
    except Exception as e:
        print(f"Erreur critique lors du chargement : {e}")
        exit()

    model.to(CONFIG['device'])
    model.eval()
    return model

# Pipeline
transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    model = load_trained_model()
    prev_probs = None
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error webcam not available.")
        return

    print("System launched: press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Padding
            pad_w, pad_h = int(w * CONFIG['padding']), int(h * CONFIG['padding'])
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(frame.shape[1], x + w + pad_w), min(frame.shape[0], y + h + pad_h)

            # Prediction
            face_roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            if face_roi.size > 0:
                pil_img = Image.fromarray(face_roi)
                tensor_img = transform(pil_img).unsqueeze(0).to(CONFIG['device'])
                
                with torch.no_grad():
                    outputs = model(tensor_img)
                    probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

                if prev_probs is None: prev_probs = probs
                else: prev_probs = CONFIG['smooth_factor'] * prev_probs + (1 - CONFIG['smooth_factor']) * probs
                
                idx = np.argmax(prev_probs)
                emotion, conf = CLASS_NAMES[idx], prev_probs[idx]
                color = COLORS.get(emotion, (255, 255, 255))

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{emotion} {conf:.0%}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Emotion Recognition (OpenCV Mode)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()