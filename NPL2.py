import argparse
import math
import os
import re
from typing import List, Tuple

import pandas as pd


MULTI_SPACE = re.compile(r'\s+')
SENTENCE_SPLIT = re.compile(r'(?<=[.!?]) +')


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.strip()
    text = MULTI_SPACE.sub(' ', text)
    return text


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sents = SENTENCE_SPLIT.split(text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents


def extractive_summary(text: str, ratio: float = 0.2, top_n: int = None) -> str:
    """Simple extractive summarizer using TF-IDF sentence scoring.

    - ratio: fraction of sentences to keep (e.g., 0.2 keeps top 20% sentences)
    - top_n: explicit number of sentences to keep (overrides ratio)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    sents = split_sentences(clean_text(text))
    if not sents:
        return ''
    if len(sents) == 1:
        return sents[0]

    if top_n is None:
        top_n = max(1, math.ceil(len(sents) * ratio))

    # Vectorize sentences
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
    X = vec.fit_transform(sents)

    # Score each sentence by the sum of tf-idf values
    sent_scores = X.sum(axis=1).A1  # convert to 1D numpy

    # pick top sentences and keep original order
    top_idx = sorted(range(len(sents)), key=lambda i: sent_scores[i], reverse=True)[:top_n]
    top_idx_sorted = sorted(top_idx)
    summary = ' '.join([sents[i] for i in top_idx_sorted])
    return summary


def abstractive_summary(texts: List[str], model_name: str = None, max_length: int = 150, min_length: int = 30):
    """Use HuggingFace transformers pipeline to produce abstractive summaries.
    Returns list of summary strings matching input order.
    """
    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError('Please install transformers and torch to use abstractive summarization: pip install transformers torch')

    if model_name:
        summarizer = pipeline('summarization', model=model_name)
    else:
        summarizer = pipeline('summarization')

    summaries = []
    batch_size = 4
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = summarizer(batch, max_length=max_length, min_length=min_length, truncation=True)
        # each out item is dict with 'summary_text'
        summaries.extend([o['summary_text'].strip() for o in out])
    return summaries



def process_single_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def summarize_dataframe(df: pd.DataFrame, text_col: str, mode: str, **kwargs) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)

    if mode == 'extractive':
        ratio = kwargs.get('ratio', 0.2)
        top_n = kwargs.get('top_n', None)
        df['summary'] = df[text_col].apply(lambda t: extractive_summary(t, ratio=ratio, top_n=top_n))
    elif mode == 'abstractive':
        model_name = kwargs.get('model_name', None)
        max_length = kwargs.get('max_length', 150)
        min_length = kwargs.get('min_length', 30)
        texts = df[text_col].tolist()
        df['summary'] = abstractive_summary(texts, model_name=model_name, max_length=max_length, min_length=min_length)
    else:
        raise ValueError('Unknown mode: ' + mode)

    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description='Text Summariser (extractive + abstractive)')
    parser.add_argument('--mode', choices=['extractive', 'abstractive'], default='extractive')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--infile', help='Path to a plain text file to summarize')
    group.add_argument('--csv', help='Path to CSV file containing texts')
    parser.add_argument('--text_col', default='text', help='Column name in CSV for text')
    parser.add_argument('--ratio', type=float, default=0.2, help='For extractive: fraction of sentences to keep')
    parser.add_argument('--top_n', type=int, help='For extractive: explicit number of sentences to keep')
    parser.add_argument('--model_name', help='For abstractive: HuggingFace model name (e.g., facebook/bart-large-cnn)')
    parser.add_argument('--max_length', type=int, default=150, help='For abstractive: max summary length')
    parser.add_argument('--min_length', type=int, default=30, help='For abstractive: min summary length')
    parser.add_argument('--out', default='summaries_out.csv', help='Output CSV (for batch) or printed for single file')

    args = parser.parse_args(argv)

    if args.infile:
        txt = process_single_text_file(args.infile)
        if args.mode == 'extractive':
            summary = extractive_summary(txt, ratio=args.ratio, top_n=args.top_n)
            print('\n===== Extractive Summary =====\n')
            print(summary)
        else:
            summaries = abstractive_summary([clean_text(txt)], model_name=args.model_name, max_length=args.max_length, min_length=args.min_length)
            print('\n===== Abstractive Summary =====\n')
            print(summaries[0])
    else:
        df = pd.read_csv(args.csv)
        if args.text_col not in df.columns:
            raise ValueError(f"CSV must contain column '{args.text_col}'")
        out_df = summarize_dataframe(df, text_col=args.text_col, mode=args.mode,
                                     ratio=args.ratio, top_n=args.top_n,
                                     model_name=args.model_name, max_length=args.max_length, min_length=args.min_length)
        out_df.to_csv(args.out, index=False)
        print('Saved summaries to', args.out)


if __name__ == '__main__':
    main()
