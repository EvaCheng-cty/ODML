import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mediapipe as mp
import cv2
import os
import pandas as pd


class KeypointRandDataset(Dataset):
    def __init__(self, json_file, image_dir, use_cnn = False, transform=None):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.annotations = data['annotations']
        self.transform = transform
        self.images = []
        self.categories = []

        image_info = data['images']
        
        for ann in self.annotations:
            image_path = os.path.join(image_dir, image_info[ann['image_id']]['file_name'])

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.categories.append(ann['category_id'])
        
        self.categories = np.array(self.categories)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1)
        self.use_cnn = use_cnn
        self.flip_prob  = 0.5
    
    def get_keypoints(self, image):
        results = self.hands.process(image)
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints.extend([[h.x, h.y, h.z] for h in hand_landmarks.landmark])
        
        return np.array(keypoints)

    def apply_random_rotation(self, keypoints, center_index=9):
        # Step 1: Translate keypoints so joint 9 becomes the origin
        center = keypoints[9]
        translated_keypoints = keypoints - center

        # Step 2: Generate random rotation angles (in radians)
        angle_x = np.random.uniform(-np.pi, np.pi)  # Random angle for X-axis rotation
        angle_y = np.random.uniform(-np.pi, np.pi)  # Random angle for Y-axis rotation
        angle_z = np.random.uniform(-np.pi, np.pi)  # Random angle for Z-axis rotation

        # Step 3: Create rotation matrices
        # Rotation matrix around X-axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        # Rotation matrix around Y-axis
        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        # Rotation matrix around Z-axis
        R_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # Combine rotation matrices (ZYX order)
        R = R_z @ R_y @ R_x

        # Step 4: Apply the rotation to the translated keypoints
        rotated_keypoints = np.dot(translated_keypoints, R.T)

        # Step 5: Translate keypoints back to the original position
        rotated_keypoints += center

        return rotated_keypoints

    def keypoints_to_image(self, keypoint):
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

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, idx):
        image = self.images[idx]
        category = self.categories[idx]

        if self.transform:
            # step1: apply transformation to image
            image = self.transform(image)
            
        # step2: get keypoints from the images
        # step3: apply random rotation to keypoints using center of hand as center
        keypoints = self.get_keypoints(image)

        if len(keypoints) == 0:
            keypoints = np.zeros((21, 3))

        keypoints = self.apply_random_rotation(keypoints)
        
        # if use cnn model, convert keypoint to 5*5 image
        if self.use_cnn:
            keypoints = self.keypoints_to_image(keypoints)

        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(category, dtype=torch.long)


class KeypointRotDataset(Dataset):
    def __init__(self, json_file, use_cnn = False, flip_prob = 0.5, rand_rot = False, transform=None):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.annotations = data['annotations']
        self.transform = transform
        self.keypoints = []
        self.flip_keypoints = []
        self.categories = []
        
        for ann in self.annotations:
            # Extract 3D keypoints (reshape them to 21 x 3)
            keypoints = np.array(ann['keypoints'])  # Assuming the keypoints are stored flat
            if keypoints.shape[0] == 0:
                continue
            self.keypoints.append(keypoints)  # Flatten to a 1D array for input
            self.categories.append(ann['category_id'])

            flip_keypoints = np.array(ann['flipped_keypoints'])
            if flip_keypoints.shape[0] == 0:
                flip_keypoints = keypoints
            self.flip_keypoints.append(flip_keypoints)
        
        self.use_cnn = use_cnn
        self.random_rot = rand_rot
        
        # Normalize the keypoints if needed
        self.keypoints = np.array(self.keypoints)
        self.categories = np.array(self.categories)
        self.flip_prob = flip_prob
    
    def update_data(self, keypoints, labels, one_indexed=True):
        self.keypoints = keypoints

        self.categories = labels

        if one_indexed:
            self.categories -= 1
    
    def set_cnn(self, use_cnn):
        self.use_cnn = use_cnn
    
    def set_rand_rot(self, rand_rot):
        self.random_rot = rand_rot

    def apply_random_rotation(self, keypoints):
        # Step 1: Translate keypoints so joint 9 becomes the origin
        center = keypoints[9]
        translated_keypoints = keypoints - center

        angle = np.pi/6

        # Step 2: Generate random rotation angles (in radians)
        angle_x = np.random.uniform(-angle, angle)  # Random angle for X-axis rotation
        angle_y = np.random.uniform(-angle, angle)  # Random angle for Y-axis rotation
        angle_z = np.random.uniform(-angle, angle)  # Random angle for Z-axis rotation

        # Step 3: Create rotation matrices
        # Rotation matrix around X-axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        # Rotation matrix around Y-axis
        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        # Rotation matrix around Z-axis
        R_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # Combine rotation matrices (ZYX order)
        R = R_z @ R_y @ R_x

        # Step 4: Apply the rotation to the translated keypoints
        rotated_keypoints = np.dot(translated_keypoints, R.T)

        # Step 5: Translate keypoints back to the original position
        rotated_keypoints += center

        return rotated_keypoints

    def apply_random_rotation_2d(self, keypoints):
        keypoints = keypoints.reshape(21, 2)
        
        # Step 1: Translate keypoints so joint 9 becomes the origin
        center = keypoints[9]  # Assuming keypoints[9] is the origin joint
        translated_keypoints = keypoints - center

        # Step 2: Generate a random rotation angle (in radians)
        angle = np.pi / 12  # Maximum rotation angle
        rotation_angle = np.random.uniform(-angle, angle)  # Random angle in range [-pi/6, pi/6]

        # Step 3: Create a 2D rotation matrix
        R = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle),  np.cos(rotation_angle)]
        ])

        # Step 4: Apply the rotation to the translated keypoints
        rotated_keypoints = np.dot(translated_keypoints, R.T)

        # Step 5: Translate keypoints back to the original position
        rotated_keypoints += center

        return rotated_keypoints

    def keypoints_to_image(self, keypoint):
        image = torch.zeros((5, 5, 2))

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
                    image[i, j, :] = torch.tensor(keypoint[index])  # Copy keypoint data (x, y, z)
        
        return image.permute(2, 0, 1)

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, idx):
        keypoint = self.keypoints[idx]
        category = self.categories[idx]

        # if np.random.rand() < self.flip_prob:
        #     keypoints = self.flip_keypoints[idx]
        
        if self.random_rot:
            keypoint = self.apply_random_rotation_2d(keypoint)
        
        # if use cnn model, convert keypoint to 5*5 image
        if self.use_cnn:
            keypoint = self.keypoints_to_image(keypoint)

        return torch.tensor(keypoint, dtype=torch.float32), torch.tensor(category, dtype=torch.long)

