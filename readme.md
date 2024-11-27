# ASL Per Frame Detection

## Data Pre-Process
The code for preprocess data is in `data_process.ipynb` which consists of:
1. Get keypoints using MediaPipe for each image
2. Flip each image and get the keypoints in the flipper image

## Dataset
The code for class `Dataset` is in `dataset.py` where the data augmentation is applied:
1. random flip
    - by certain probability, we use the keypoints from the flipped image
    - If the flipped image has no keypoints, use the original one
2. keypoint rotation
    - added a random $[-\frac{\pi}{6}, \frac{\pi}{6}]$ rotation to x,y,z axis of the keypoints

## Training
The training script for models are in `train_val_cnn_aug.ipynb`

## Inference
To inference, run
```
python runMediaPipe_CNN.py
```
and use model weight `cnn_model_new.pth`

## Updates and Experiments
Compared to previous version, I have added the new Dataset class that apply the random transformation to keypoints and the flipped keypoints for images.

Something I have tried are:
1. Use Mediapipe during training to obtain keypoints, in this way we can first apply image transformation, but it is super slow
2. For random rotation I used to use range $[-\pi,\pi]$, but then the model confuses letters like O and E
3. The current hyperparameters are the ones that does best so far. I have tried:
    - remove second CNN layer
    - for second CNN layer use channel 16
    - 2 and 3 hidden layers
    - batch size 32
    - hidden size 64, 256
    - epochs: 50, 100, 200
    - lr: 0.01, 0.005 along with same weight decay
