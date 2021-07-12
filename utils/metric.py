from difflib import SequenceMatcher
from collections import defaultdict


def compute_f1(n_correct, n_pred, n_gold):
    """ Compute P, R, F1 scores. """
    precision = n_correct / (n_pred + 1e-10)  # avoid divide-zero
    recall = n_correct / (n_gold + 1e-10)
    f1 = (2 * precision * recall) / (precision + recall + 1e-10)

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return results


def is_string_match(s1, s2):
    def gestalt_matching_score(s1, s2):
        m = SequenceMatcher(None, s1, s2)
        return m.ratio()

    MATCHING_THRESHOLD = 0.85

    return gestalt_matching_score(s1, s2) >= MATCHING_THRESHOLD


def predicate_evaluate(examples, predictions):
    gold_predicates = {
        example.unique_id: example.all_predicates
        for example in examples
    }

    n_correct = 0
    gold_total, pred_total = 0, 0
    matched_gold = {}
    for unique_id, predicates in gold_predicates.items():
        # Note that gold may have the same predicates
        gold_set = gold_predicates[unique_id]
        pred_set = [pred for pred, start, end in predictions[unique_id]]
        gold_total += len(gold_set)
        pred_total += len(pred_set)

        pred_matched_gold = []
        for pred in pred_set:
            matched = None
            for gold in gold_set:
                if is_string_match(pred, gold):
                    matched = gold
                    n_correct += 1
                    break
            if matched:
                gold_set.remove(matched)
            pred_matched_gold.append(matched)
        matched_gold[unique_id] = pred_matched_gold

    results = compute_f1(n_correct, pred_total, gold_total)
    results["matched_gold"] = matched_gold
    return results


def entity_evaluate(example_gold_triples, example_pred_triples):
    """ Evaluate entity prediction given predicates. """
    subj_correct, obj_correct, all_correct = 0, 0, 0
    gold_total, pred_total = 0, 0
    example_pred_result = {}
    for example_id, gold_triples in example_gold_triples.items():
        pred_triples = example_pred_triples[example_id]
        gold_total += len(gold_triples)
        pred_total += len(pred_triples)
        pred_result = []
        for gold_triple, pred_triple in zip(gold_triples, pred_triples):
            assert gold_triple[1] == pred_triple[1]
            correct_type = "none"
            if is_string_match(gold_triple[0], pred_triple[0]):
                subj_correct += 1
                correct_type = "subject"
            if is_string_match(gold_triple[2], pred_triple[2]):
                obj_correct += 1
                correct_type = "object"
            if is_string_match(gold_triple[0], pred_triple[0]) and \
                is_string_match(gold_triple[2], pred_triple[2]):
                all_correct += 1
                correct_type = "all"
            pred_result.append(correct_type)
        example_pred_result[example_id] = pred_result

    assert gold_total == pred_total
    result = {
        "subject_accuracy": subj_correct / (gold_total + 1e-10),
        "object_accuracy": obj_correct / (gold_total + 1e-10),
        "all_accuracy": all_correct / (gold_total + 1e-10),
        "example_pred_result": example_pred_result
    }
    return result


def triple_evaluate(gold_triples, pred_triples):
    n_correct = 0
    gold_total, pred_total = 0, 0
    for sentence, gold in gold_triples.items():
        pred = pred_triples[sentence]
        gold_set = set(gold)
        pred_set = set(pred)

        # Exactly same triples are correct
        n_correct += len(gold_set & pred_set)
        gold_total += len(gold)
        pred_total += len(pred)

    results = compute_f1(n_correct, pred_total, gold_total)
    return results


def saoke_evaluate(all_gold_triples, all_pred_triples):
    """ OIE triple evaluation as in SAOKE paper.
        But no optimal matching between gold-pred.
    """

    def is_triple_match(t1, t2):
        """ If two triples are considered the same. """
        if not len(t1) == len(t2):
            return False

        # matching as a whole string
        s1, s2 = " ".join(t1), " ".join(t2)
        if is_string_match(s1, s2):
            return True

        # matching each item
        for s1, s2 in zip(t1, t2):
            if not is_string_match(s1, s2):
                return False
        return True

    sentence_results = {}
    correct_total = 0
    gold_total, pred_total = 0, 0
    for unique_id, gold_triples in all_gold_triples.items():
        pred_triples = all_pred_triples[unique_id]

        # local P, R, F1
        n_correct = 0
        n_gold, n_pred = 0, 0

        n_pred += len(pred_triples)
        n_gold += len(gold_triples)

        # match gold_triple with firstly founded pred_triple
        # TODO: linear assignment match scipy.optimize.linear_sum_assignment
        matched_triples = []
        for gold_triple in gold_triples:
            for pred_triple in pred_triples:
                if pred_triple in matched_triples:
                    continue
                if is_triple_match(pred_triple, gold_triple):
                    n_correct += 1
                    matched_triples.append(pred_triple)
                    break

        results = compute_f1(n_correct, n_pred, n_gold)
        sentence_results[unique_id] = results

        correct_total += n_correct
        pred_total += n_pred
        gold_total += n_gold

    print("num of (correct / prediction / gold) triples:", correct_total, pred_total, gold_total)
    final_results = compute_f1(correct_total, pred_total, gold_total)
    return final_results, sentence_results
