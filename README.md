# P1_text-classification

Text classification using an **LSTM** model. The model is trained on a CSV dataset (text + label), with vocabulary-based vectorization (word-to-index) and one-hot encoding of the labels.

## Project Structure

```text
.
├── main.py
├── model/
│   ├── __init__.py
│   └── LSTMmodel.py
├── data/
│   └── bodo_poopy.csv
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

The script expects a CSV file with at least two columns:

* `text`: the text to classify
* `Label`: the associated class

## Model Architecture

`LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.5)`

* Embedding layer
* Multi-layer LSTM (`batch_first=True`)
* Final linear layer using the last hidden state

