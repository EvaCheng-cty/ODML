import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import json


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

def keypoints_to_image(keypoint):
    image = torch.zeros((5, 5, 3))

    # Define the mapping as a 5x5 grid
    mapping = [
        [4, 8, 12, 16, 20],
        [3, 7, 11, 15, 19],
        [2, 6, 10, 14, 18],
        [1, 5, 9, 13, 17],
        [-1, -1, 0, -1, -1]  # -1 for x (zero-padding entries), 0 for index 0
    ]

    for i in range(5):
        for j in range(5):
            index = mapping[i][j]
            if index != -1:  # If not a padded zero
                image[i, j, :] = torch.Tensor(keypoint[index])  # Copy keypoint data (x, y, z)
    
    return image.permute(2, 0, 1)


def parse_class(pred):
    if pred == 0:
        return "None"
    elif 1 <= pred <= 26:
        return chr(ord('A') + pred - 1)


class CNNModel(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(CNNModel, self).__init__()
        # self.layers = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  # Output: 16 x 5 x 5
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Output: 32 x 5 x 5
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(32 * 5 * 5, hidden_dim),  # Flattened input size
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, num_classes)
        # )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),  # Output: 16 x 5 x 5
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Output: 32 x 5 x 5
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, hidden_dim),  # Flattened input size
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),  # Flattened input size
            # nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
       

    def forward(self, x):
        return F.softmax(self.layers(x), dim = 1) 



input_dim = 63
hidden_dim = 256    
num_classes = 26

# Reinitialize and load the model
loaded_model = CNNModel(hidden_dim, num_classes)
loaded_model.load_state_dict(torch.load("cnn_model_2d.pth"))
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

            input = keypoints_to_image(np.array([[h.x, h.y, h.z] for h in hand_landmarks.landmark])).unsqueeze(0)
            with torch.no_grad():
                pred = loaded_model(input[:, :2])
                print(pred)
                score, pred = torch.max(pred, 1)
                if score < 0.5:
                    pred = 0
            
            pred_char = parse_class(pred)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  # White color (B, G, R)
            thickness = 2
            cv2.putText(image, pred_char+" " +str(score.item()), (image.shape[1] - 150, 30), font, font_scale, color, thickness)  # Top-right corner




    # Display the image
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
