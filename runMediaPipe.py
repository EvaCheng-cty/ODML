import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import json


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)


def parse_class(pred):
    if pred == 0:
        return "None"
    elif 1 <= pred <= 26:
        return chr(ord('A') + pred - 1)


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)


input_dim = 63
hidden_dim = 256
num_classes = 26

# Reinitialize and load the model
loaded_model = MLPModel(input_dim, hidden_dim, num_classes)
loaded_model.load_state_dict(torch.load("mlp_model.pth"))
loaded_model.eval()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            input = torch.flatten(torch.Tensor([[h.x, h.y, h.z] for h in hand_landmarks.landmark]))
            with torch.no_grad():
                pred = torch.argmax(loaded_model(input)).item()
            
            pred_char = parse_class(pred)
            print(pred_char)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  # White color (B, G, R)
            thickness = 2
            cv2.putText(image, pred_char, (image.shape[1] - 150, 30), font, font_scale, color, thickness)  # Top-right corner




    # Display the image
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
