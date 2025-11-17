import re
import argparse
import os
import sys
from typing import List, Tuple

import pandas as pd

# Text cleaning patterns
URL_RE = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_RE = re.compile(r'<.*?>')
NON_ALPHANUMERIC_RE = re.compile(r'[^0-9a-zA-Z\s]')
MULTI_SPACE_RE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """Basic cleaning for movie reviews."""
    if not isinstance(text, str):
        return ''
    t = text
    t = t.replace('\n', ' ').replace('\r', ' ')
    t = HTML_TAG_RE.sub(' ', t)
    t = URL_RE.sub(' ', t)
    t = NON_ALPHANUMERIC_RE.sub(' ', t)
    t = MULTI_SPACE_RE.sub(' ', t)
    t = t.strip().lower()
    return t


def train_classical(df: pd.DataFrame, text_col: str = 'review', label_col: str = 'label') -> Tuple[object, pd.DataFrame]:
    """Train a TF-IDF + LogisticRegression model and return (pipeline, eval_df).

    eval_df contains columns: text, true_label, pred_label, pred_proba
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import numpy as np
    except Exception as e:
        print('Install scikit-learn to use classical pipeline: pip install scikit-learn')
        raise

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns: {text_col}, {label_col}")

    df = df.copy()
    df['clean_text'] = df[text_col].astype(str).apply(clean_text)

    X = df['clean_text']
    y = df[label_col]

    # simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=30000)),
        ('clf', LogisticRegression(max_iter=2000))
    ])

    print('Training classical TF-IDF + LogisticRegression model...')
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    try:
        probs = pipe.predict_proba(X_test).max(axis=1)
    except Exception:
        probs = [None] * len(preds)

    print('Accuracy on test set:', accuracy_score(y_test, preds))
    print('Classification report:\n', classification_report(y_test, preds))

    eval_df = pd.DataFrame({
        'text': X_test,
        'true_label': y_test,
        'pred_label': preds,
        'pred_proba': probs
    })

    return pipe, eval_df


def transformer_predict(texts: List[str], model_name: str = None) -> List[dict]:
    """Use HuggingFace transformers pipeline for sentiment analysis.

    Returns list of dicts with keys 'label' and 'score'.
    """
    try:
        from transformers import pipeline
    except Exception:
        print('Install transformers and torch to use transformer predictions: pip install transformers torch')
        raise

    # default model depends on transformers; specifying model_name is recommended
    if model_name:
        nlp = pipeline('sentiment-analysis', model=model_name)
    else:
        nlp = pipeline('sentiment-analysis')

    batch_size = 16
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = nlp(batch)
        outputs.extend(out)
    return outputs


def save_sklearn_model(pipe, path: str):
    try:
        import joblib
    except Exception:
        raise RuntimeError('Install joblib (usually comes with scikit-learn)')
    joblib.dump(pipe, path)
    print('Saved model to', path)


def load_sklearn_model(path: str):
    try:
        import joblib
    except Exception:
        raise RuntimeError('Install joblib (usually comes with scikit-learn)')
    return joblib.load(path)



def main(argv=None):
    parser = argparse.ArgumentParser(description='Movie Review Sentiment Analysis')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--csv', required=True, help='Path to CSV file with reviews')
    parser.add_argument('--text_col', default='review', help='Column name for review text')
    parser.add_argument('--label_col', default='label', help='Column name for labels (for training)')
    parser.add_argument('--model_out', help='Path to save trained classical model (joblib .pkl)')
    parser.add_argument('--model_in', help='Path to load classical model for prediction')
    parser.add_argument('--use_transformer', action='store_true', help='Use HuggingFace transformer pipeline for prediction')
    parser.add_argument('--model_name', help='HuggingFace model name for transformer pipeline (optional)')
    parser.add_argument('--out', default='predictions.csv', help='CSV file to save predictions')

    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)

    if args.mode == 'train':
        # training classical model
        pipe, eval_df = train_classical(df, text_col=args.text_col, label_col=args.label_col)
        if args.model_out:
            save_sklearn_model(pipe, args.model_out)
        # save evaluation table
        eval_out = 'evaluation_results.csv'
        eval_df.to_csv(eval_out, index=False)
        print('Saved evaluation results to', eval_out)

    elif args.mode == 'predict':
        # ensure we have text col
        if args.text_col not in df.columns:
            raise ValueError(f"CSV must contain column '{args.text_col}'")

        texts = df[args.text_col].astype(str).apply(clean_text).tolist()

        if args.use_transformer:
            print('Running transformer-based predictions...')
            out = transformer_predict(texts, model_name=args.model_name)
            labels = [o.get('label') for o in out]
            scores = [o.get('score') for o in out]
            df['pred_label'] = labels
            df['pred_score'] = scores
            df.to_csv(args.out, index=False)
            print('Saved transformer predictions to', args.out)
        else:
            # classical model predict
            if not args.model_in:
                raise ValueError('For classical prediction please provide --model_in (path to saved model)')
            pipe = load_sklearn_model(args.model_in)
            preds = pipe.predict(texts)
            try:
                probs = pipe.predict_proba(texts).max(axis=1)
            except Exception:
                probs = [None] * len(preds)
            df['pred_label'] = preds
            df['pred_score'] = probs
            df.to_csv(args.out, index=False)
            print('Saved classical predictions to', args.out)


if __name__ == '__main__':
    main()
