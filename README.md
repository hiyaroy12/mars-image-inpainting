## Mars image inpainting:
This is a project page for inpainting the MARS surface image captured by the HiRISE camera on the Mars Reconnaissance Orbiter. 

### Introduction:
Sophisticated imaging instruments on-board spacecraft orbiting different planets and planetary bodies in this solar system, enable humans to discover and visualize the unknown. However, these planetary surface images suffer from some missing pixel regions, which could not be captured by the spacecraft onboard cameras because of some technical limitations. In this work, we try to inpaint these missing pixels of the planetary images using modality-specific regression models that were trained with clusters of different images with similar histogram distribution on the experimental dataset. Filling in missing data via image inpainting enables downstream scientific analysis such as the analysis of morphological features on the planetary surface - e.g., craters and their sizes, central peaks, interior structure, etc|in comparison with other planetary bodies. Here, we use the grayscale version of Mars orbital images captured by the HiRISE (High-Resolution Imaging Science Experiment) camera on the Mars Reconnaissance Orbiter (MRO) for the experimental purpose. The results show that our method can fill in the missing pixels existing on the Mars surface image with good visual and perceptual quality and improved PSNR values. Detailed description of the system can be found in our [paper.](https://www.dropbox.com/s/fkb148zciaj69r6/elsarticle_final.pdf?dl=0)

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
This code is inspired from [EdgeConnect](https://github.com/knazeri/edge-connect)