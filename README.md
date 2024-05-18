# Cancer Detection using Vision Transformer (ViT) - Malignant vs. Normal

Welcome to the Cancer Detection using Vision Transformer repository. This project focuses on using Vision Transformers (ViT) to classify cancerous tissue samples into two categories: malignant and normal. The dataset used in this project is private, and therefore, not included in this repository.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Cancer diagnosis through histopathological images is a crucial task in medical imaging. Accurate classification of these images can significantly aid in early detection and treatment. This repository leverages Vision Transformers (ViT), a state-of-the-art deep learning model, to perform binary classification on a dataset of histopathological images.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation

To set up this project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/cancer-detection-vit.git
    cd cancer-detection-vit
    ```

2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

This project uses a private dataset containing histopathological images of cancerous and normal tissue samples. Due to privacy concerns and data sharing restrictions, the dataset is not provided in this repository. However, the code is designed to be flexible and can be adapted to work with similar datasets.

Ensure your dataset is structured in the following format:

```
dataset/
    train/
        malignant/
        normal/
    val/
        malignant/
        normal/
    test/
        malignant/
        normal/
```

## Model Architecture

The Vision Transformer (ViT) model utilized in this project is based on the "google/vit-base-patch16-224-in21k" checkpoint from Hugging Face. ViT models use self-attention mechanisms to process image patches, making them highly effective for image classification tasks.

### Loading the Model

The model and image processor are loaded as follows:

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForImageClassification.from_pretrained(checkpoint)
```

## Training

The training script trains the ViT model on the provided dataset. It is configured to handle data loading, preprocessing, and model training. To train the model, execute:

```bash
python train.py --data_dir path/to/dataset --output_dir path/to/save/model
```

The training script includes parameters for hyperparameter tuning, batch size, learning rate, and other training configurations. Adjust these parameters as necessary.

## Evaluation Metrics

The model's performance is evaluated using several metrics:

- **Accuracy:** The ratio of correctly predicted instances to the total instances.
- **Precision:** The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall:** The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1 Score:** The weighted average of Precision and Recall.
- **Matthews Correlation Coefficient (MCC):** A measure of the quality of binary classifications.

### Compute Metrics Function

The following function computes these metrics:

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    mcc = matthews_corrcoef(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc
    }
```

## Usage

To use the trained model for inference on new images, run:

```python
python predict.py --image_path path/to/image --model_dir path/to/saved/model
```

This script will preprocess the input image, run the model, and output the predicted class (malignant or normal).

## Results

After training and evaluating the model, the results, including the evaluation metrics and confusion matrix, will be saved in the specified output directory. Detailed logs and model checkpoints are also saved for further analysis.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to thank the open-source community for their invaluable tools and libraries that made this project possible. Special thanks to the Hugging Face team for their transformers library and to all contributors who provided helpful feedback and suggestions.

---

For any questions or issues, please open an issue in this repository. Happy coding!
