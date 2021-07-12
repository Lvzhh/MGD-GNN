import spacy
from spacy.tokens import Doc
from spacy.symbols import POS, TAG, DEP, HEAD

import numpy as np

"""
copy from Span_OIE for producing predicate candidates
"""


def make_span_candidates(l):
    candidates = []
    for i in range(l):
        for j in range(i,l):
            candidates.append([i,j])
    return candidates


def span_candidates_filter(candidates, max_len):
    new_candidates = []
    for candidate in candidates:
        if candidate[1] - candidate[0] + 1 <= max_len:
            new_candidates.append(candidate)
    return new_candidates


def syntax_check(sent_spacy, span):
    parent = []
    if span[1] - span[0] == 0:
        return True
    for i in range(span[0], span[1] + 1):
        parent.append(sent_spacy[i].head.i)
    for i in range(span[0], span[1] + 1):
        if ((parent[i - span[0]] >= span[0]) and (parent[i - span[0]] <= span[1])) or (i in parent):
            pass
        else:
            return False
    return True


def pred_span_candidates_filter(sent_spacy, candidates, max_len):
    candidates = span_candidates_filter(candidates, max_len)
    new_candidates = []
    for candidate in candidates:
        if syntax_check(sent_spacy, candidate):
            new_candidates.append(candidate)
    return new_candidates


def LTP_model_wrapper(nlp, example):
    words = example["ltp_segments"]
    spaces = [False] * len(words)
    tags = example["ltp_pos"]
    deps = [rel for _, _, rel in example["ltp_dependency"]]
    # relative index for spacy
    heads = [head - i if head != 0 else 0 for i, head, _ in example["ltp_dependency"]]

    tags = [nlp.vocab.strings.add(label) for label in tags]
    deps = [nlp.vocab.strings.add(label) for label in deps]

    attrs = [TAG, DEP, HEAD]
    arr = np.array(list(zip(tags, deps, heads)), dtype="uint64")
    doc = Doc(nlp.vocab, words=words, spaces=spaces).from_array(attrs, arr)

    assert example["ltp_segments"] == [token.text for token in doc]
    assert example["ltp_pos"] == [token.tag_ for token in doc]
    assert [rel for _, _, rel in example["ltp_dependency"]] == [token.dep_ for token in doc]
    assert [head - 1 if head != 0 else i - 1 for i, head, _ in example["ltp_dependency"]] == [token.head.i for token in doc]
    return doc