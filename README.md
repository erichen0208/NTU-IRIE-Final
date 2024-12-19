# Legal Document Retrieval System

This project is designed to retrieve relevant legal documents based on user queries using a BERT-based model specifically trained for Chinese legal texts.

## Setup

To set up the environment and install the necessary dependencies, follow these steps:

1. Create a new conda environment:
    ```sh
    conda create --name irie python=3.9
    ```

2. Activate the conda environment:
    ```sh
    conda activate irie
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### To run the main script and retrieve relevant legal documents, use the following command:
```python
CUDA_VISIBLE_DEVICES=2 python main.py --mode inference
```
The mode has 'llm', 'bert', 'finetune', 'generate', 'inference', 'test', 'test_embeddings', choose one to use
