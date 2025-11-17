import re
import argparse
import os
import sys
import string
from typing import List

import pandas as pd


EMOJI_PATTERN = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_HASHTAG_PATTERN = re.compile(r'[@#]\w+')
NON_ALPHANUMERIC = re.compile(r'[^0-9a-zA-Z\s]')
MULTI_SPACE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """Clean a single comment text."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # remove urls
    text = URL_PATTERN.sub(' ', text)
    # remove mentions and hashtags
    text = MENTION_HASHTAG_PATTERN.sub(' ', text)
    # remove emojis
    text = EMOJI_PATTERN.sub(' ', text)
    # remove punctuation and non-alphanumeric (keep spaces)
    text = NON_ALPHANUMERIC.sub(' ', text)
    # lower
    text = text.lower()
    # collapse spaces
    text = MULTI_SPACE.sub(' ', text).strip()
    return text


def load_comments_from_csv(path: str, text_col: str = 'comment_text') -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"CSV must contain a '{text_col}' column. Found: {list(df.columns)}")
    df = df.copy()
    df['clean_text'] = df[text_col].astype(str).apply(clean_text)
    return df



def train_classical_model(df: pd.DataFrame, text_col: str = 'clean_text', label_col: str = 'label'):
    """Train a TF-IDF + LogisticRegression sentiment model.

    Expects df to contain text_col and label_col. Labels can be binary (0/1) or multiclass.
    Returns trained pipeline and the test DataFrame with predictions.
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import classification_report, accuracy_score
    except Exception as e:
        print("Please install scikit-learn to use classical ML mode: pip install scikit-learn")
        raise

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")

    X = df[text_col].fillna('')
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print('Accuracy on test set:', accuracy_score(y_test, preds))
    print('Classification report:\n', classification_report(y_test, preds))

    results = X_test.to_frame().copy()
    results['true_label'] = y_test.values
    results['pred_label'] = preds

    return pipe, results



def transformer_predict(texts: List[str], model_name: str = None):
    """Use HuggingFace transformers pipeline for sentiment analysis.

    model_name: optional — if None the pipeline default model will be used (can be platform dependent).
    Returns list of dicts (label, score) per input.
    """
    try:
        from transformers import pipeline
    except Exception:
        print("Please install transformers (and torch) to use the transformer option: pip install transformers torch")
        raise

    if model_name:
        nlp = pipeline('sentiment-analysis', model=model_name)
    else:
        nlp = pipeline('sentiment-analysis')

    # run in batches
    batch_size = 16
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = nlp(batch)
        outputs.extend(out)
    return outputs


def save_model_sklearn(pipe, path: str):
    import joblib
    joblib.dump(pipe, path)


def load_model_sklearn(path: str):
    import joblib
    return joblib.load(path)



def main(argv=None):
    parser = argparse.ArgumentParser(description='Instagram Comment Sentiment Analysis')
    parser.add_argument('--mode', choices=['train', 'predict'], default='predict', help='train or predict')
    parser.add_argument('--csv', required=True, help='Path to CSV with comments (must have comment_text column)')
    parser.add_argument('--text_col', default='comment_text', help='Name of the text column in the CSV')
    parser.add_argument('--label_col', default='label', help='Name of the label column (for training)')
    parser.add_argument('--use_transformer', action='store_true', help='Use transformer pipeline for prediction (no training)')
    parser.add_argument('--model_out', help='Path to save trained sklearn model (for train mode)')
    parser.add_argument('--model_in', help='Path to a saved sklearn model (for predict mode)')
    parser.add_argument('--model_name', help='HuggingFace model name for transformer pipeline (optional)')
    parser.add_argument('--out', default='predictions_out.csv', help='CSV to save predictions')

    args = parser.parse_args(argv)

    # Load comments
    df = load_comments_from_csv(args.csv, text_col=args.text_col)

    if args.mode == 'train':
        if args.use_transformer:
            print('Transformer mode does not support training in this script. Use classical mode or fine-tune separately.')
            sys.exit(1)
        # Train classical model
        pipe, results = train_classical_model(df, text_col='clean_text', label_col=args.label_col)
        if args.model_out:
            save_model_sklearn(pipe, args.model_out)
            print('Saved model to', args.model_out)
        # Save test results
        results.to_csv(args.out, index=False)
        print('Saved evaluation results to', args.out)

    elif args.mode == 'predict':
        texts = df['clean_text'].fillna('').tolist()

        if args.use_transformer:
            print('Running transformer-based predictions...')
            out = transformer_predict(texts, model_name=args.model_name)
            # convert output to DataFrame
            labels = [o.get('label') for o in out]
            scores = [o.get('score') for o in out]
            df['pred_label'] = labels
            df['pred_score'] = scores
            df.to_csv(args.out, index=False)
            print('Saved predictions to', args.out)
        else:
            # classical model predict
            if args.model_in is None:
                raise ValueError('Please provide --model_in for classical model prediction (trained sklearn pipeline)')
            pipe = load_model_sklearn(args.model_in)
            preds = pipe.predict(texts)
            # If pipe supports predict_proba
            try:
                probs = pipe.predict_proba(texts).max(axis=1)
            except Exception:
                probs = [None] * len(preds)
            df['pred_label'] = preds
            df['pred_score'] = probs
            df.to_csv(args.out, index=False)
            print('Saved predictions to', args.out)


if __name__ == '__main__':
    main()
