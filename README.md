## Mars image inpainting:
This is a project page for inpainting the MARS surface image captured by the HiRISE camera on the Mars Reconnaissance Orbiter. 

### Introduction:
Orbital images of planetary surfaces are collected in long strips using the on-board imaging instrument as the orbiting spacecraft sweeps over the planets or planetary bodies. However, these images suffer from missing pixel regions or artifacts, because the cameras on-board the orbiting spacecraft fail to capture some region on the planet's surface due to different technical reasons such as the swath width settings of the on-board cameras, malfunction of the onboard sensors or other electronic reasons. This greatly reduce the usability of the captured data which can be solved by predicting such missing pixels using image inpainting algorithms. In this work, we employ a deep convolutional neural network based image inpainting algorithm to predict missing pixel in a context-aware fashion. The main motivation behind inpainting the planetary images is to recognize and classify different interesting landmarks such as craters, sand dunes, slope streaks, impact ejecta etc on the planetary surface in a better way. Here, we use the grayscale version of Mars orbital images captured by the HiRISE (High-Resolution Imaging Science Experiment) camera on the Mars Reconnaissance Orbiter (MRO) for the experimental purpose. Experimental results show that our method can fill in the missing pixels on the Mars surface image with good visual and perceptual quality and improved PSNR values and reflects that such missing pixel predictions via image inpainting is effective in improving the classification accuracy of different morphological features in planetary images. [Paper (coming soon)]

## Prerequisites: 
- Python 3
- PyTorch 1.0
- NVIDIA GPU + CUDA cuDNN
- some dependencies like cv2, numpy etc. 

## Dataset: 
- Dataset can be downloaded here: [Mars orbital image (HiRISE)labeled data set](https://zenodo.org/record/2538136#.XYjEuZMzagR)

## Getting started:

### Data preparation:
- In Martian orbital images input images have multi-modal intensity distribution. We tackled this problem by clustering images with similar intensity distribution and then training regression models having expertise in restoring missing pixels in the images with that particular intensity distribution. 
- We found the optimal number of clusters=5 by carrying out Knee point analysis.
- Therefore while training and testing we train the model separately for different clusters and together using all images at once.  
- The clean and corrupted filenames belonging to each cluster can be found in data/clean_files_cluster_{}.txt and data/corrupted_files_cluster_{}.txt
- You can download the mask files extracted from the corrupted images here: https://www.dropbox.com/s/flecpseziuwkyx7/mask_train_test.npz?dl=0 and copy it under data directory.


### 1) Training
To train the model, create a config.yaml file similar to the example config file and copy it under CEEC directory.
- You can train the model for different clusters "n" (0-4 in our case) by using:
```bash
python train.py --dataset mars_hirise --cluster "n" --l1_adv 
```
- You can train the model for all images together (i.e. not dividing them into clusters) by using:
```bash
python train.py --dataset mars_hirise --cluster "all" --l1_adv 
```

### 2) Testing
- To test the model for different clusters "n" (0-4 in our case) use:
```bash
python test.py --dataset mars_hirise --cluster "0" --l1_adv
```
- To test the model using all images together (i.e. not dividing them into clusters) use:
```bash
python test.py --dataset mars_hirise --cluster "all" --l1_adv
```

### 3) Evaluation:
To evaluate the model, first run the model in test mode against your validation set and save the results on disk. 
Then run metrics.py to evaluate the model using PSNR, SSIM and Mean Absolute Error:
```bash
python metrics.py --data-path [path to validation set] --output-path [path to model output]
```

## Acknowledgments
This code is inspired from [EdgeConnect.](https://github.com/knazeri/edge-connect)