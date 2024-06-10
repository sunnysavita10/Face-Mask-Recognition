# Mask Recognition using VGG16

This repository contains a Python script for recognizing whether a person is wearing a mask using the VGG16 model and Keras. The script involves data preprocessing, model training, and evaluation.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset used for this project is organized in the following structure:


Ensure the images are placed in their respective folders.

```
mask_recgnition/
├── training/
│   ├── with_mask/
│   └── without_mask/
```

## Installation
To run this project, you need to have Python and the following libraries installed:
- keras
- numpy
- matplotlib
- glob

You can install the required libraries using the following command:
```
pip install keras numpy matplotlib
```
## Usage
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/mask-recognition.git
    ```
2. Navigate to the project directory:
    ```bash
    cd mask-recognition
    ```
3. Ensure your dataset is structured as described in the [Dataset](#dataset) section.
4. Run the script:
    ```bash
    python mask_recognition.py
    ```

## Model Architecture
The script uses the VGG16 model with the following modifications:
- The input layer is resized to (224, 224, 3).
- The top layers of VGG16 are excluded.
- A Flatten layer is added.
- A Dense layer with softmax activation is added for classification.

## Training
The model is trained using the images from the training set. Data augmentation techniques such as rescaling, shear transformation, zooming, and horizontal flipping are applied to improve the model's performance.

Example training configuration:
```python
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('mask_recgnition/training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

r = model.fit_generator(training_set, epochs=5,
  steps_per_epoch=len(training_set))

## Evaluation
After training, the model's loss and accuracy are plotted for both training and validation sets:
```python
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
```

