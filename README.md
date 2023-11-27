# path_mnist_2023_erdos
Biomedical Image Classification group project - Erdos Institute Bootcamp 2023  
Group Members:  
Boyang Wu  
Jason Nguyen  
Shahinde Dogruer  
Shashwata Moitra  



# NCT-CRC-HE-100K Classifcation with EfficientNet

This repository contains the code for training a Convolutional Neural Network (CNN) on the CRC (Colorectal Cancer) dataset using the EfficientNet architecture. The training script employs cosine learning rate scheduling and supports training on multiple GPUs.

## Requirements (needs to verify this later)

- Python 3.x
- PyTorch (version 2.0.1 or later)
- torchvision
- efficientnet_pytorch
- PIL (Pillow)
- tqdm

## Installation

1. Clone this repository to your local machine.
2. Install the required Python packages. It's recommended to use a virtual environment:

   ```bash
   pip install torch torchvision efficientnet_pytorch Pillow tqdm
   ```
## Usage

To train the model using NCT-CRC-HE-100K dataset, use the example below to run the training script from command line:



  ```bash
  python training_CRC_cosineLR.py --train_data_dir "path_to_training_data" \
                                  --val_data_dir "path_to_validation_data" \
                                  --model_name "efficientnet-b0" \
                                  --output_dir "path_to_output_directory" \
                                  --resized_dim 224 \
                                  --learning_rate 1e-3
  
  ```
### Arguments
```--train_data_dir```: The directory containing the training data.

```--val_data_dir```: The directory containing the validation data.

```--model_name```: The EfficientNet model version to use (e.g., efficientnet-b0).

```--output_dir```: The directory where the training logs and model checkpoints will be saved.

```--resized_dim```: The dimensions to which the input images will be resized (default is 224x224 pixels).

```--learning_rate```: The learning rate for the Adam optimizer.

## Input Data Structure
The training and validation data directories should have the following structure (as provided by authors of NCT-CRC-HE-100K dataset (insert citation link here):
  ```bash
  data_directory/
    ADI/
        image1.tif
        image2.tif
        ...
    BACK/
        image1.tif
        ...
    ...
  ```

## Training Output
The script will save model checkpoints and a training log CSV file in the specified output directory. The CSV file contains epoch-wise training and validation accuracy and loss.

## Model Evaluation

(to be added)
