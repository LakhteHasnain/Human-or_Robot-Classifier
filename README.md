# Human or Robot - Image Classification

A begineer friendly computer vision project that classifies images as either **human portraits** or **robots** using deep learning. Built with fastai and ResNet18, this project demonstrates how accessible modern machine learning has become.

## Overview

This project fine-tunes a pretrained neural network to distinguish between human portraits and robot images. It showcases the power of transfer learning and how you can build a functional image classifier in just a few minutes using entirely free resources.

## Features

- 🤖 **Automated Image Collection**: Uses DuckDuckGo to search and download training images
- 🧠 **Transfer Learning**: Leverages ResNet18 pretrained model for fast training
- 📊 **Data Validation**: Automatically verifies and cleans corrupted images
- 🎯 **High Accuracy**: Achieves reliable classification with minimal training data
- 📈 **Easy Predictions**: Simple API to classify new images

## Requirements

- Python 3.7+
- fastai
- duckduckgo_search (>= 6.2)
- fastcore
- fastdownload
- matplotlib
- PIL (Pillow)

## Installation

```bash
# Install required packages
pip install -Uqq fastai 'duckduckgo_search>=6.2' ddgs fastcore fastdownload matplotlib pillow
```

## Quick Start

### 1. Download Training Images

```python
from ddgs import DDGS
from fastcore.all import *

def search_images(keywords, max_images=200):
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')

# Search for human portraits and robot images
human_urls = search_images('human portrait photos', max_images=5)
robot_urls = search_images('robots photos', max_images=5)
```

### 2. Organize and Prepare Data

```python
from fastai.vision.all import *
from pathlib import Path
import time

searches = 'human potraits', 'robots'
path = Path('human_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    time.sleep(5)
    resize_images(path/o, max_size=400, dest=path/o)

# Verify and remove corrupted images
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
```

### 3. Create DataLoaders

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
```

### 4. Train the Model

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

### 5. Make Predictions

```python
from fastai.vision.all import PILImage

is_human, _, probs = learn.predict(PILImage.create('human_1.jpg'))
print(f"This is: {is_human}.")
print(f"Probability it's a human: {probs[0]:.4f}")
```

## How It Works

1. **Image Collection**: Uses DuckDuckGo API to search for and download images of human portraits and robots
2. **Data Preparation**: Resizes images to 192x192 pixels and validates image integrity
3. **Model Architecture**: Uses ResNet18 (a pretrained convolutional neural network) as the backbone
4. **Transfer Learning**: Fine-tunes the model on your specific classification task for just 3 epochs
5. **Classification**: Makes predictions on new images with confidence scores

## DataBlock Parameters Explained

- **blocks=(ImageBlock, CategoryBlock)**: Defines inputs as images and outputs as categories
- **get_items=get_image_files**: Retrieves all image files from the dataset path
- **splitter=RandomSplitter(valid_pct=0.2, seed=42)**: Splits data into 80% training and 20% validation
- **get_y=parent_label**: Uses folder names as labels (human potraits or robots)
- **item_tfms=[Resize(192, method='squish')]**: Resizes images to 192x192 pixels

## Results

The model achieves high accuracy in distinguishing between human portraits and robot images. Example output:

```
This is: human.
Probability it's a human: 0.9876
```

## Project Structure

```
human_or_not/
├── human potraits/     # Human portrait images
├── robots/             # Robot images
└── model files         # Trained model weights
```

## Notes

- This project was inspired by an XKCD joke from 2015 about how difficult it was to create systems that could recognize humans
- The entire workflow is designed to run on free resources (Kaggle notebooks, free APIs)
- Training typically takes just a few minutes on modern hardware

## Future Improvements

- Add more image categories (e.g., animals, objects)
- Implement model persistence (save/load trained models)
- Create a web interface for easy predictions
- Expand training data for better generalization
- Add confidence thresholds for predictions

## License

MIT License - Feel free to use this project for educational and commercial purposes.

## Acknowledgments

- Built with [fastai](https://www.fast.ai/) - A deep learning library
- Image search powered by [DuckDuckGo](https://duckduckgo.com/)
- Inspired by the fastai course and community

---

**Happy classifying!** 🎉
