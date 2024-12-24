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

Firstly, we do continue pre-training for bert model:
```python
CUDA_VISIBLE_DEVICES=0 python main.py --mode bert
```
Second, we fine tune the bert model on the training data:
```python
CUDA_VISIBLE_DEVICES=0 python main.py --mode finetune
```
Third, we generate the law embeddings:
```python
CUDA_VISIBLE_DEVICES=0 python main.py --mode generate
```
Lastly, we inference the test data and output submission.csv:
```python
CUDA_VISIBLE_DEVICES=0 python main.py --mode inference
```
