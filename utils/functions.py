import numpy as np
from tqdm import tqdm
from preprocess import DataProcessor
from feature_converter import get_segment_char_span


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, gaz, char_alphabet, label_alphabet, gaz_alphabet, number_normalized, max_sent_length, inst_type="entity"):
    processor = DataProcessor()
    processor.read_examples_from_json(input_file)
    if inst_type == "entity":
        examples = processor.get_entity_examples()
    else:  # relation
        examples = processor.get_relation_examples()

    instance_texts = []
    instance_ids = []
    cut_num = 0
    for example in tqdm(examples):
        segment_char_span = get_segment_char_span(example)
        dependency_head = [head_index for word_index, head_index, dep_rel in example.dependency]
        dependency_rel = [dep_rel for word_index, head_index, dep_rel in example.dependency]

        chars = [char for char in example.sentence]
        if number_normalized:
            pass
        char_ids = [char_alphabet.get_index(char) for char in chars]

        # label and predicate_position
        if inst_type == "entity":
            labels = [example.subj_span, example.obj_span]
            label_ids = labels

            start, end = example.pred_span  # [start, end]
            pos = list(range(-start, 0)) + [0] * (end - start + 1) + list(range(1, len(chars) - end))
            pos = [idx + max_sent_length for idx in pos]
            assert len(pos) == len(chars)

            # TODO: better ways to handle longer sentences during **both** training and test :)
            pos = [0 if idx < 0 else
                idx if idx < 2 * max_sent_length else
                2 * max_sent_length - 1 for idx in pos]
        else:
            labels = [[0] * len(chars), [0] * len(chars)]
            for pred_start, pred_end in example.pred_spans:
                labels[0][pred_start] = 1.0
                labels[1][pred_end] = 1.0
            label_ids = labels
            pos = None

        if True or ((max_sent_length < 0) or (len(chars) <= max_sent_length)) and (len(chars) > 0):
            gazs = []
            gaz_ids = []
            s_length = len(chars)
            for idx in range(s_length):
                matched_list = gaz.enumerateMatchList(chars[idx:])
                matched_length = [len(a) for a in matched_list]
                gazs.append(matched_list)
                matched_id = [gaz_alphabet.get_index(entity) for entity in matched_list]
                if matched_id:
                    gaz_ids.append([matched_id, matched_length])
                else:
                    gaz_ids.append([])
            example_info = {
                "unique_id": example.unique_id,
                "orig_id": example.orig_id,
                "sentence": example.sentence,
                "all_predicates": example.all_predicates,
                "subject": example.subject,
                "predicate": example.predicate,
                "object": example.object
            }
            instance_texts.append([chars, gazs, labels, example_info])
            instance_ids.append([char_ids, gaz_ids, label_ids, pos, segment_char_span, dependency_head, dependency_rel])
        elif max_sent_length < len(chars):
            cut_num += 1
    return instance_texts, instance_ids, cut_num


def build_pretrain_embedding(embedding_path, alphabet, skip_first_row=False, separator=" ", embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path, skip_first_row, separator)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for alph, index in alphabet.iteritems():
        if alph in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph])
            else:
                pretrain_emb[index, :] = embedd_dict[alph]
            perfect_match += 1
        elif alph.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[alph.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding: %s\n     pretrain num:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
    embedding_path, pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path, skip_first_row=False, separator=" "):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        i = 0
        j = 0
        for line in tqdm(file):
            if i == 0:
                i = i + 1
                if skip_first_row:
                    _ = line.strip()
                    continue
            j = j+1
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(separator)
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if embedd_dim + 1 == len(tokens):
                    embedd = np.empty([1, embedd_dim])
                    embedd[:] = tokens[1:]
                    embedd_dict[tokens[0]] = embedd
                else:
                    continue
    return embedd_dict, embedd_dim
