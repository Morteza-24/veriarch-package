"""Parses a Java monolith into per-class records.

This is the front door of the pipeline: it produces the raw material
consumed by structural.py (call graph -> S^str), semantic.py (tokens ->
embeddings), and descriptors.py (tokens/comments -> capability
descriptors d_k), matching the parsing stage shared with Mo2oM.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set

import javalang


@dataclass
class ClassInfo:
    name: str
    file_path: str
    tokens: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    calls: Set[str] = field(default_factory=set)


def parse_repository(repo_path: str) -> Dict[str, ClassInfo]:
    """Walk repo_path and parse every .java file into a ClassInfo record.

    Files that fail to parse (syntax errors, partial snippets, generated
    code) are skipped rather than aborting the whole run.
    """
    classes: Dict[str, ClassInfo] = {}
    for root, _, files in os.walk(repo_path):
        for fname in files:
            if not fname.endswith(".java"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    src = f.read()
            except OSError:
                continue
            try:
                tree = javalang.parse.parse(src)
            except Exception:
                continue

            comments = _extract_comments(src)
            tokens = _tokenize(src)

            for _, node in tree.filter(javalang.tree.ClassDeclaration):
                info = classes.setdefault(
                    node.name, ClassInfo(name=node.name, file_path=fpath)
                )
                info.comments.extend(comments)
                info.tokens.extend(tokens)
                info.calls |= _extract_calls(node)
    return classes


def _extract_comments(src: str) -> List[str]:
    comments = []
    i = 0
    n = len(src)
    while i < n - 1:
        if src[i:i + 2] == "//":
            j = src.find("\n", i)
            j = j if j != -1 else n
            comments.append(src[i:j])
            i = j
        elif src[i:i + 2] == "/*":
            j = src.find("*/", i)
            j = (j + 2) if j != -1 else n
            comments.append(src[i:j])
            i = j
        else:
            i += 1
    return comments


def _tokenize(src: str) -> List[str]:
    try:
        return [tok.value for tok in javalang.tokenizer.tokenize(src)]
    except Exception:
        return []


def _extract_calls(node) -> Set[str]:
    calls: Set[str] = set()
    for _, sub in node.filter(javalang.tree.MethodInvocation):
        if getattr(sub, "qualifier", None):
            calls.add(sub.qualifier)
    for _, sub in node.filter(javalang.tree.ClassCreator):
        sub_type = getattr(sub, "type", None)
        if sub_type is not None and hasattr(sub_type, "name"):
            calls.add(sub_type.name)
    return calls
