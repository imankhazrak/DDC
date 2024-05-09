
# ResNet Deep Convolutional Classifier Projects

This repository contains a series of Jupyter notebooks that demonstrate the application and tuning of ResNet models for different image classification tasks using Deep Convolutional Classifiers. The notebooks cover a range of configurations and dataset scenarios.

## Notebooks Overview

### ResnetDCC_Iman.ipynb
This notebook outlines the application of a custom ResNet model tailored for a specific image classification task. The model setup, training, and evaluation steps are detailed, focusing on achieving high accuracy on a challenging dataset.

### ResnetDCC_Amazon_dslr_resnet50.ipynb
This notebook applies a ResNet-50 model to classify images from an Amazon product dataset captured using DSLR cameras. It explores data preprocessing, model training, fine-tuning, and performance evaluation specific to high-resolution images.

### ResnetDCC_Amazon_dslr_resnet101.ipynb
An extension of the ResNet-50 approach, this notebook employs the ResNet-101 architecture to handle more complex image patterns from the same Amazon DSLR dataset, aiming to improve the accuracy and robustness of the classification results.

### ResnetDCC_Amazon_webcam_resnet50.ipynb
Focusing on images taken from webcams, this notebook adapts the ResNet-50 model to classify lower-quality images from the Amazon product dataset, addressing unique challenges such as varying lighting and lower resolution.

### ResnetDCC_Amazon_webcam_resnet101.ipynb
Similar to the DSLR notebooks but for webcam images, this notebook leverages the deeper ResNet-101 model to enhance classification performance, dealing with the intricacies of less consistent image quality.

### ResnetDCC_cuda.ipynb
This notebook demonstrates how to leverage CUDA for accelerating the computation of ResNet models on GPU hardware, focusing on setup, configuration, and optimization to harness the full potential of GPU processing.

### ResnetDCC_cuda_v2.ipynb
An advanced version of the previous CUDA notebook, providing deeper insights into CUDA optimizations, advanced model training techniques, and performance tuning to maximize efficiency and speed in processing.

## Setup and Installation
To run these notebooks, ensure you have the following installed:
- Python 3.8 or newer
- Jupyter Notebook or JupyterLab
- Required Python libraries: `torch`, `torchvision`, `numpy`, `matplotlib`



## Usage
Navigate to the directory containing the notebooks and start Jupyter:
```
jupyter notebook
```
Open the desired notebook and follow the instructions within to run the experiments.

## Contributing
Contributions to improve the notebooks or extend the models are welcome. Please fork the repository and submit a pull request with your changes.

## License
Specify the license under which the code is released, such as MIT or GPL.
```

You can adapt this README file by adding more details specific to each notebook or any additional sections that you think are necessary for users or contributors to understand the scope and functionality of your project.