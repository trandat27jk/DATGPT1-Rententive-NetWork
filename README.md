This repository provides an implementation of the Retentive Network (RetNet), as proposed in the paper:
"Retentive Network: A Successor to Transformer for Large Language Models".

-RetNet introduces an innovative architecture that combines the strengths of recurrent and attention mechanisms to efficiently handle long-context sequences, surpassing Transformers in scalability and memory efficiency. This repository includes:

A modular and extensible PyTorch implementation of RetNet.
Tools for training on datasets like the Fineweb dataset.
GPU-parallelized training for scalability.
Features for handling long-context text generation with token dropping.


-Highlights
Efficient Long-Context Processing: Leverages retention mechanisms for better scalability with long input sequences.
Transformer Successor: Implements advancements designed to address Transformers' limitations in memory and efficiency.
Customizability: Easily adaptable to various datasets and tasks.
Parallel GPU Training: Fully optimized for multi-GPU environments.

-Features
Retention Mechanism: RetNet retains contextual information with minimal computational overhead.
Token Dropping: Includes functionality to selectively drop tokens during generation, enhancing efficiency without compromising performance.
Advanced PyTorch Training Pipeline: Incorporates modern practices like mixed-precision training, gradient accumulation, and distributed training.


-Requirements

Python 3.8+

PyTorch 2.0+

CUDA-compatible GPUs for parallel training


Install dependencies:

```
pip install -r requirements.txt
```
Usage
Training on Fineweb Dataset
Prepare the Dataset
Download and preprocess the Fineweb dataset:

bash
Sao chép mã
python data/fineweb.py
Train the Model
Run the training script with multi-GPU support:
```
torchrun --standalone --nproc_per_node=8 train.py
```
samples from RetNet after 1k3 epochs
```
Rank 0 Samples
"Hello, I'm a language model, and I've got to think like some big one.”
"Hello, I'm a language model, I do not have the same level of language, a level that’s been developed with a standard of Spanish as..."
"Hello, I'm a language model, I'm here for you," he said. That's why his students are so busy teaching the language."
Rank 1 Samples
"Hello, I'm a language model, for any future.<|endoftext|>What is Unecdur?"
"Hello, I'm a language model, a model that describes the way a human sees and learns; I understand that there is lots of mathematical logic and logic that..."
```
Model Architecture
Retention Mechanism: Combines explicit memory and recurrent processing for handling extensive contexts.
Feedforward and Self-Attention: RetNet minimizes reliance on standard attention mechanisms, focusing on efficient token retention.
Scalability: Optimized for large language models, making it a strong successor to Transformers.
References
Paper: Retentive Network: A Successor to Transformer for Large Language Models
Authors: Yi Tay et al.
Future Work
Experimenting with larger datasets and tasks (e.g., OpenWebText, C4).
Incorporating advanced token dropping strategies.
Extending support to other frameworks like JAX.
