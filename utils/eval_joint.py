import os
import json
import argparse
import collections
from collections import defaultdict, OrderedDict
from preprocess import DataProcessor
from metric import triple_evaluate, saoke_evaluate


def final_result_evaluate(args):
    gold_triples = dict()
    pred_triples = dict()

    # Get gold triples
    eval_processor = DataProcessor()
    eval_processor.read_examples_from_json(args.gold_file)
    for example in eval_processor.examples:
        unique_id = example.unique_id
        gold_triples[unique_id] = []  # sentence without spo_triples or manual_labels
        for spo_span in example.spo_spans:
            pred_span = spo_span["predicate"]
            subj_span, obj_span = spo_span["subject"], spo_span["object"]
            spo_triple = tuple(example.sentence[start:end + 1] for start, end in [subj_span, pred_span, obj_span])
            gold_triples[unique_id].append(spo_triple)

    # Get prediction triples from saved file
    with open(args.pred_file, "r") as file:
        pred_examples = json.load(file)

    for example in pred_examples:
        unique_id = example["unique_id"]
        pred_triples[unique_id] = [
            (subj, pred, obj)
            for subj, pred, obj in example["predictions"]
        ]

    # Add missing examples to pred_triples
    # If no relation predicted, then no pred examples for entity
    for unique_id in gold_triples:
        if unique_id not in pred_triples:
            pred_triples[unique_id] = []

    assert gold_triples.keys() == pred_triples.keys()
    final_results, sentence_results = saoke_evaluate(gold_triples, pred_triples)

    triple_predictions = []
    # Examples with their extracted triples
    for example in eval_processor.examples:
        unique_id = example.unique_id
        sentence = example.sentence
        sent_gold_triples = gold_triples[unique_id]
        sent_pred_triples = pred_triples[unique_id]
        results = sentence_results[unique_id]

        example_with_triples = collections.OrderedDict(
            [
                ("unique_id", unique_id),
                ("sentence", sentence),
                ("gold_triples", [f"({', '.join(t)})" for t in sent_gold_triples]),
                ("prediction_triples", [f"({', '.join(t)})" for t in sent_pred_triples]),
                ("precision", results["precision"]),
                ("recall", results["recall"]),
                ("f1", results["f1"])
            ]
        )
        triple_predictions.append(example_with_triples)

    output_prediction_file = os.path.join(args.output_dir, "triple_predictions.json")
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(triple_predictions, ensure_ascii=False, indent=4) + "\n")

    for key, value in final_results.items():
        print("eval_{}: {}".format(key, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Directory containing intermediate output files & to write results.")
    parser.add_argument("--gold_file", default=None, type=str)
    parser.add_argument("--pred_file", default=None, type=str, help="Entity prediction result file.")

    args = parser.parse_args()
    if args.gold_file is not None:
        final_result_evaluate(args)
