# Integrated framework for fusing radiogenomics and metabolic modelling via deep learning in ovarian cancer

This code is part of the repository contains the code and data to reproduce the results presented in the paper: N. Eftekhari, A. Occhipinti, C. Angione, "Integrated framework for fusing radiogenomics and metabolic modelling via deep learning in ovarian cancer".



<img style="width: 720px; alignment: center" src="Data/Fig1-pipeline.png">



In the first step of the feature generation process, we used a U-shaped convolutional neural network and a set of labelled data to segment tumour patches for patients without labelling information (Fig.1.A). The reason behind segmentation was to determine the contour of the tumour and extract the tissue pattern of the tumour area to integrate with the omics . In our dataset, most of the patient's CT images included full body scans; some also had post-operative CT scans in which the tumour area was removed through the surgery. Therefore, we removed the post-operation scans and selected the CT scans of interest.
As a result, 119 patients with CT images were selected. A training set of 50 patients was labelled by the radiologist in our team (Dr Saha), utilising the standard clinical process for patients at the NHS North Tees. We trained two versions of the U-Net models with different hyper-parameters and different numbers of CT slides. The first version introduced the whole CT scans for each patient as input data (some patients had full-body CT scans, including lungs and breasts). In the second version, we used the female reproductive system scans with the same number of scans for each patient and trained the model with 20 scans per patient. 
In the post-processing step, after extracting the tumour-predicted area for each patient, we applied the region-growing method to each scan. We removed the noise to segment the 3D tumour patch for the patients without labelling information.


## Steps to run the proposed U-net models for ovarian cancer prediction
Training: Run [train/train.py.py]

Prediction: Run [train/ovarian_unet_predict.py]
