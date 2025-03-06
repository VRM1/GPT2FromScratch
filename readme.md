# GPT-2 Implementation: Language Models are Unsupervised Multitask Learners

This repository contains an implementation of the GPT-2 language model based on the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) by Radford et al. The implementation is built by adapting the Transformer architecture from the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper.

**Note**: This might not be an exact 1:1 reproduction of the GPT-2 model, but the core essential components are mostly the same.

The code is inspired by and builds upon the excellent Transformer implementation by [Yu-Hsiang Huang](https://github.com/jadore801120/attention-is-all-you-need-pytorch).

## Features

- Implementation of GPT-2 architecture using PyTorch
- Support for training on multiple tasks without fine-tuning (zero-shot learning)
- Text generation with temperature, top-k, and nucleus (top-p) sampling
- Training on translation and sentiment analysis tasks
- Interactive testing and automatic evaluation
- Checkpoint saving and training resumption

## Project Structure

```
.
├── transformer/                  # Core transformer module
│   ├── __init__.py
│   ├── gpt2.py                   # Main GPT-2 model implementation
│   ├── Layers.py                 # Implementation of the decoder layer
│   ├── SubLayers.py              # Attention and feed-forward implementations
│   └── Modules.py                # Basic components like attention mechanisms
├── data_preparation.py           # Dataset preparation for multitask learning
├── main.py                       # Training script
├── test_model.py                 # Interactive testing script
├── eval_model.py                 # Automatic evaluation script
├── config.yml                    # Configuration file
└── README.md                     # This file
```

## Key Differences from Transformer to GPT-2

- **Architecture**: GPT-2 uses only the decoder part of the Transformer
- **Layer Normalization**: Applied before each sub-layer (pre-norm), rather than after (post-norm)
- **Positional Embeddings**: Uses learned positional embeddings instead of sinusoidal
- **Larger Scale**: Designed to scale to much larger model sizes (up to 1.5B parameters)
- **Activation Function**: Uses GELU instead of ReLU in feed-forward layers
- **Byte-Pair Encoding**: Uses a larger vocabulary (50,257 tokens) with BPE tokenization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt2-implementation.git
cd gpt2-implementation
```

2. Install required packages using the provided requirements file:
```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda env create -f requirements.yml
conda activate gpt2-env
```

3. Download NLTK resources (for BLEU calculation):
```python
import nltk
nltk.download('punkt')
```

## Usage

### Preparing the Data

```bash
python main.py --prepare_data
```

### Training the Model

```bash
python main.py --config config.yml
```

### Resuming Training from a Checkpoint

```bash
python main.py --resume --resume_checkpoint /path/to/checkpoint.pt
```

### Interactive Testing

```bash
python test_model.py --model_path /path/to/model.pt
```

Example usage in interactive mode:
```
Choose task (1=Translation, 2=Sentiment, 3=Custom): 1
Enter French text: Bonjour, comment allez-vous?
```

### Automatic Evaluation

```bash
python eval_model.py --model_path /path/to/model.pt --output evaluation_results.json
```

## Configuration

The `config.yml` file contains parameters for model architecture, training, and evaluation.

Key configuration sections:
- **model**: Architecture parameters (dimensions, layers, heads, etc.)
- **training**: Training parameters (batch size, epochs, learning rate, etc.)
- **data**: Dataset parameters (sequence length, tokenizer, etc.)
- **output**: Output parameters (log intervals, checkpoint saving, etc.)
- **test**: Testing parameters (model path, generation settings, etc.)

## Zero-Shot Learning

In the original paper "Language Models are Unsupervised Multitask Learners," the authors demonstrated that GPT-2 can perform multiple tasks without explicit supervision by simply pretraining on a large, diverse corpus of web text (WebText).

This implementation takes a more focused approach due to computational constraints. Instead of training on a massive general corpus, we simulate the multitask learning capability by:

1. Combining two specific datasets (translation and sentiment analysis)
2. Formatting the examples with task-specific prompts
3. Training the model to recognize these patterns during pretraining

At inference time, the model can complete these patterns without task-specific fine-tuning, showing how language models can perform multiple tasks with appropriate prompting, even when trained on a smaller, more focused dataset.

## Results

When trained on sufficient data, the model can:
1. Translate French text to English
2. Classify sentiment of reviews
3. Generate coherent text continuations for custom prompts

All without task-specific fine-tuning, demonstrating the multitask learning capabilities of autoregressive language models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Implementation inspired by [Yu-Hsiang Huang's Transformer implementation](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- Architecture based on the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) by OpenAI
