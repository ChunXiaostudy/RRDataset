# Image Classification with Vision Language Models

This repository contains code for evaluating various vision language models on the task of distinguishing between real and AI-generated images. The code supports any OpenAI API-compatible endpoint.

## Features

- Support for multiple image types (original, transfer, redigital)
- In-context learning with example images
- Comprehensive evaluation metrics (overall accuracy, real accuracy, fake accuracy)
- Real-time result saving
- Image optimization for token efficiency

## Project Structure

```
test/
├── original_real_images/
├── original_ai_images/
├── transfer_real_images/
├── transfer_ai_images/
├── redigital_real_images/
└── redigital_ai_images/
```

## Usage

1. Configure your API settings:
```python
api_key = "YOUR_API_KEY"
base_url = "YOUR_API_ENDPOINT"
```

2. Run basic classification:
```bash
python image_classification.py
```

3. Run in-context learning classification:
```bash
python run_incontext_test.py --model "your-model-name"
```

## Supported Models

The code has been tested with various vision language models that follow the OpenAI API format, including but not limited to:
- GPT-4V
- Claude 3
- GLM-4V
- Gemini Pro
- And more...

## Requirements

- Python 3.8+
- openai
- Pillow
- tqdm
- pandas

## Installation

```bash
pip install openai pillow tqdm pandas
```
