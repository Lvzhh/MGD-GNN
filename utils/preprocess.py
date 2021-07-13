import os
import re
import json
import unicodedata
from json import JSONDecodeError
from tqdm import tqdm
from collections import OrderedDict


class RelationInputExample(object):
    """ A single sentence with all annotated predicate spans. """

    def __init__(self, unique_id, sentence, pred_spans, segments, dependency):
        self.unique_id = unique_id
        self.sentence = sentence
        self.pred_spans = pred_spans
        self.segments = segments
        self.dependency = dependency

    def __getattr__(self, name):
        if name == "all_predicates":
            all_predicates = [self.sentence[pred_start:pred_end + 1]
                for pred_start, pred_end in self.pred_spans]
            return all_predicates


class EntityInputExample(object):
    # TODO: pickle.dump copy.deepcopy(entity_example) issue
    """A single training/test example for subj && obj span prediction."""

    def __init__(self, unique_id, orig_id, sentence, subj_span, pred_span, obj_span,
                 answer_type="subject", segments=None, dependency=None):
        self.unique_id = unique_id
        # one sentence may have multiple EntityExample(s)
        # orig_id are original example uniqud_id for merging predictions
        self.orig_id = orig_id
        self.sentence = sentence
        self.subj_span = subj_span
        self.pred_span = pred_span
        self.obj_span = obj_span
        # whether subject or object as answer in SQuAD metric
        self.answer_type = answer_type
        self.segments = segments
        self.dependency = dependency

    def __getattr__(self, name):
        """ Wrapper for using SQuAD metric. """
        if name == "qas_id":
            return self.unique_id
        elif name == "answers":
            answers = []
            if self.answer_type == "subject":
                answers.append({"text": self.subject})
            else:
                answers.append({"text": self.object})
            return answers
        elif name in ["subject", "predicate", "object"]:
            prefix = name[:3] if name == "object" else name[:4]
            start, end = getattr(self, prefix + "_span")
            return self.sentence[start:end+1] if (start, end) != (-1, -1) else ""

    def __deepcopy__(self, memo):
        """ Avoid TypeError when using copy.deepcopy() with __getattr__ defined. """
        if id(self) in memo:
            return memo[id(self)]
        else:
            copied = type(self)(self.unique_id, self.orig_id, self.sentence,
                                self.subj_span, self.pred_span, self.obj_span,
                                self.answer_type, self.segments, self.dependency)
            memo[id(self)] = copied
            return copied

    def to_dict(self):
        example_dict = OrderedDict(
            [
                ("unique_id", self.unique_id),
                ("sentence", self.sentence),
                ("subject", self.subject),
                ("object", self.object),
                ("predicate", self.predicate)
            ]
        )
        return example_dict


class InputExample(object):
    """A single sentence with all annotated (s,p,o)."""

    def __init__(self, unique_id, sentence, spo_list, spo_spans,
                 segments=None, pos=None, dependency=None):
        self.unique_id = unique_id
        self.sentence = sentence
        self.spo_list = spo_list
        self.spo_spans = spo_spans
        # segments, pos, dependency for chinese
        self.segments = segments
        self.pos = pos  # part of speech, not position
        self.dependency = dependency

    def to_entity_examples(self, start_unique_id):
        entity_examples = []
        for i, spo_span in enumerate(self.spo_spans):
            pred_span = spo_span["predicate"]
            subj_span, obj_span = spo_span["subject"], spo_span["object"]
            entity_examples.append(EntityInputExample(
                                    unique_id=start_unique_id + i,
                                    orig_id=self.unique_id,
                                    sentence=self.sentence,
                                    subj_span=subj_span,
                                    pred_span=pred_span,
                                    obj_span=obj_span,
                                    segments=self.segments,
                                    dependency=self.dependency
                                    ))
        return entity_examples

    def to_relation_examples(self, unique_id):
        relation_examples = [RelationInputExample(
            unique_id=unique_id,
            sentence=self.sentence,
            pred_spans=[spo_span["predicate"] for spo_span in self.spo_spans],
            segments=self.segments,
            dependency=self.dependency
        )]
        return relation_examples


class DataProcessor(object):

    def __init__(self):
        self.examples = None

    def get_entity_examples(self, start_unique_id=0):
        """ Get examples for entity span prediction. """
        entity_examples = []
        for example in self.examples:
            entity_examples += example.to_entity_examples(start_unique_id)
            if len(entity_examples) > 0:
                start_unique_id = entity_examples[-1].unique_id + 1

        return entity_examples

    def get_relation_examples(self):
        """ Get examples for relation span prediction. """
        relation_examples = []
        for example in self.examples:
            # one example corresponds to one relation_example, so the same id
            unique_id = example.unique_id
            relation_examples += example.to_relation_examples(unique_id)
        return relation_examples

    def read_examples_from_json(self, data_file_or_data):
        data = data_file_or_data
        if isinstance(data_file_or_data, str):
            with open(data_file_or_data, "r") as file:
                data = json.load(file)

        n_facts = 0
        examples = []
        for unique_id, example in enumerate(data):
            # if example already has a unique_id field
            unique_id = example.get("unique_id", unique_id)
            sentence = example["sentence"]
            spo_list = example.get("spo_list", [])
            spo_spans = example.get("spo_spans", [])
            segments = example.get("ltp_segments", None)
            pos = example.get("ltp_pos", None)
            dependency = example.get("ltp_dependency", None)

            examples.append(InputExample(
                unique_id=unique_id,
                sentence=sentence,
                spo_list=spo_list,
                spo_spans=spo_spans,
                segments=segments,
                pos=pos,
                dependency=dependency
            ))
            n_facts += len(spo_spans)
        n_sentences = len(examples)
        print("#sentences:", n_sentences)
        print("#facts(spo triples):", n_facts)
        self.examples = examples


def get_segment_char_span(example):
    sentence = example.sentence
    segment_char_span = []
    char_start = 0
    while char_start < len(sentence):
        segment = example.segments[len(segment_char_span)]
        # remove additional whitespaces from segment result
        segment = "".join(char for char in segment if not _is_whitespace(char))
        matched = False
        for char_end in range(char_start, len(sentence)):
            # remove whitespaces and special chars from sentence
            sentence_span = "".join(char for char in sentence[char_start:char_end + 1]
                                    if not _is_non_segment(char))
            if sentence_span == segment:
                # assign whitespaces and special chars to previous segments
                while char_end + 1 < len(sentence) and _is_non_segment(sentence[char_end + 1]):
                    char_end += 1
                matched = True
                segment_char_span.append([char_start, char_end])
                char_start = char_end + 1
                break
        assert matched, (sentence, segment)

    # make segments cover all chars
    segment_char_span[0][0] = 0
    segment_char_span[-1][-1] = len(sentence) - 1
    for idx in range(1, len(segment_char_span)):
        if segment_char_span[idx - 1][-1] + 1 < segment_char_span[idx][0]:
            segment_char_span[idx][0] = segment_char_span[idx - 1][-1] + 1
    return segment_char_span


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_non_segment(char):
    """ If a char is not included in segmentation result. """
    cp = ord(char)
    return cp == 0 or cp == 0xfffd or _is_control(char) or \
            _is_whitespace(char)
