# Fake News Detection with a Bidirectional LSTM (PyTorch)

An intermediate NLP project: classify news articles as **Fake** or **Real** using
a from-scratch text pipeline (vocabulary building, padding, embeddings) feeding
a Bidirectional LSTM classifier.

## Dataset

**Kaggle: "Fake and Real News Dataset"**
Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

- ~44,000 news articles, split into `Fake.csv` and `True.csv`
- Columns: `title`, `text`, `subject`, `date`
- Balanced binary classification task (label 0 = fake, 1 = real)

### Download instructions
```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/ --unzip
```
After this you should have `data/Fake.csv` and `data/True.csv`.


## Folder structure
```
project1_fake_news_bilstm/
├── data/                
├── train.py              
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python train.py --data_dir data --epochs 8 --batch_size 64
```

The script will:
1. Load and merge `Fake.csv` / `True.csv`, build labels.
2. Clean text (lowercase, strip punctuation/URLs).
3. Build a vocabulary (min frequency threshold) and convert text to padded
   integer sequences.
4. Train a BiLSTM classifier with an embedding layer, dropout, and a
   fully-connected classification head.
5. Print per-epoch train/val loss & accuracy, then a final test report
   (accuracy, precision, recall, F1, confusion matrix).
6. Save the trained model to `model.pt` and the vocabulary to `vocab.json`.


## Future Enhancements
- Add an attention layer over LSTM outputs and visualize which words the
  model attends to.
- Swap the trainable embedding for pretrained GloVe (300d) and compare.
- Fine-tune DistilBERT on the same split and compare metrics + inference time.
- Deploy as a small Gradio/Streamlit app where a user pastes an article.
