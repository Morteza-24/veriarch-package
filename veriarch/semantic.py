"""Semantic feature extraction.

Computes unit-normalized semantic embeddings e-hat_k (Sec. 3.1 /
Sec. 4.1 of methodology.tex), reusing Mo2oM's choice of UniXcoder when
transformers/torch are available in the environment, and falling back to
a TF-IDF + SVD projection so the rest of the pipeline stays runnable
without a GPU or model download.
"""

from typing import Dict, List

import numpy as np


def compute_semantic_embeddings(
    classes: Dict[str, "object"], names: List[str], dim: int = 256
) -> np.ndarray:
    try:
        return _unixcoder_embeddings(classes, names, dim)
    except Exception:
        return _tfidf_embeddings(classes, names, dim)


def _unixcoder_embeddings(classes, names, dim):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = AutoModel.from_pretrained("microsoft/unixcoder-base")
    model.eval()

    embeddings = np.zeros((len(names), model.config.hidden_size))
    with torch.no_grad():
        for i, name in enumerate(names):
            info = classes[name]
            text = " ".join(info.tokens[:512]) + " " + " ".join(info.comments[:50])
            inputs = tokenizer(
                text, truncation=True, max_length=512, return_tensors="pt"
            )
            out = model(**inputs)
            embeddings[i] = out.last_hidden_state.mean(dim=1).squeeze(0).numpy()

    embeddings = _normalize_rows(embeddings)
    if embeddings.shape[1] != dim:
        embeddings = _project(embeddings, dim)
    return embeddings


def _tfidf_embeddings(classes, names, dim):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [
        " ".join(classes[n].tokens) + " " + " ".join(classes[n].comments)
        for n in names
    ]
    vectorizer = TfidfVectorizer(max_features=5000)
    x = vectorizer.fit_transform(docs)

    n_components = max(1, min(dim, x.shape[0] - 1, x.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components)
    reduced = svd.fit_transform(x)
    return _normalize_rows(reduced)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _project(mat: np.ndarray, dim: int) -> np.ndarray:
    from sklearn.random_projection import GaussianRandomProjection

    proj = GaussianRandomProjection(n_components=dim)
    return _normalize_rows(proj.fit_transform(mat))
