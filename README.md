# Integrated framework for fusing radiogenomics and metabolic modelling via deep learning in ovarian cancer

This code is part of the repository contains the code and data to reproduce the results presented in the paper: N. Eftekhari, A. Saha, S. Verma, G. Zampieri, S. Sawan, A. Occhipinti, C. Angione, "Fusing imaging and metabolic modelling via multimodal deep learning in ovarian cancer".



<img style="width: 720px; alignment: center" src="Images/Fig1-pipeline.png">

# Installation
*** Clone the Repository
git clone https://github.com/Angione-Lab/Integrated-framework-for-fusing-radiogenomics-and-metabolic-modelling-via-deep-learning.git
cd Integrated-framework-for-fusing-radiogenomics-and-metabolic-modelling-via-deep-learning

*** Install Dependencies
pip install -r requirements.txt


# Image Segmentation
conda create -n ovarian_unet_py37 python=3.7
conda activate ovarian_unet_py37

conda install tensorflow-gpu==2.1
in case got error:
conda install tensorflow-estimator=2.1.0

conda install tensorflow-mkl==2.1 (cpu version)
pip install nibabel SimpleItk matplotlib tqdm


