import sys
sys.path.insert(0, "../chinese-openie")

from utils.data import Data
from utils.batchify import batchify
from utils.config import get_args
from utils.metric import get_ner_fmeasure
from model.bilstm_gat_crf import BLSTM_GAT_CRF, PredicateExtractionModel, PredicateSpanScoreModel
import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import time
import random
import gc

import json
import collections
from tqdm import tqdm, trange
from metric import entity_evaluate, predicate_evaluate

import sys
sys.path.insert(0, "../chinese-openie")
from feature_converter import get_segment_char_span
from utils.pred_candidates import LTP_model_wrapper, make_span_candidates, pred_span_candidates_filter
import spacy


def data_initialization(args):
    data_stored_directory = args.data_stored_directory
    file = data_stored_directory + args.dataset_name + "_dataset.dset"
    if os.path.exists(file) and not args.refresh:
        data = load_data_setting(data_stored_directory, args.dataset_name)
    else:
        data = Data()
        data.dataset_name = args.dataset_name
        data.norm_char_emb = args.norm_char_emb
        data.norm_gaz_emb = args.norm_gaz_emb
        data.number_normalized = args.number_normalized
        data.max_sentence_length = args.max_sentence_length
        data.build_gaz_file(args.gaz_file)
        data.generate_instance(args.train_file, "train", False, inst_type="relation")
        data.generate_instance(args.dev_file, "dev", inst_type="relation")
        data.generate_instance(args.test_file, "test", inst_type="relation")
        data.build_char_pretrain_emb(args.char_embedding_path)
        data.build_gaz_pretrain_emb(args.gaz_file)
        data.fix_alphabet()
        # data.get_tag_scheme()
        save_data_setting(data, data_stored_directory)
    return data


def save_data_setting(data, data_stored_directory):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(data_stored_directory):
        os.makedirs(data_stored_directory)
    dataset_saved_name = data_stored_directory + data.dataset_name +"_dataset.dset"
    with open(dataset_saved_name, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", dataset_saved_name)


def load_data_setting(data_stored_directory, name):
    dataset_saved_name = data_stored_directory + name + "_dataset.dset"
    with open(dataset_saved_name, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", dataset_saved_name)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def _get_relation_spans(seq_probs, seq_len):
    THRESHOLD = 0.5
    start_probs, end_probs = np.split(seq_probs, 2, axis=1)
    start_probs, end_probs = start_probs.squeeze(1)[:seq_len], end_probs.squeeze(1)[:seq_len]
    start_positions = [i for i, prob in enumerate(start_probs) if prob > THRESHOLD]
    end_positions = [i for i, prob in enumerate(end_probs) if prob > THRESHOLD]

    relation_spans = []
    for char_start in start_positions:
        legal_char_end = [idx for idx in end_positions if idx >= char_start]
        if len(legal_char_end) == 0:
            continue
        char_end = min(legal_char_end)
        relation_spans.append([char_start, char_end])
    return relation_spans


def recover_label(pred_variable, batch_pred_candidates, mask_variable, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, 4): pred tag result
            gold_variable (batch_size, 4): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = permutate_batch(pred_variable, word_recover)
    batch_pred_candidates = permutate_batch(batch_pred_candidates, word_recover)
    mask_variable = mask_variable[word_recover]
    mask = mask_variable.cpu().data.numpy()
    batch_size = mask.shape[0]

    pred_label = []
    for idx in range(batch_size):
        seq_logits = pred_variable[idx].detach().cpu()
        _, indices = torch.max(seq_logits, dim=1)
        seq_len = np.sum(mask[idx])
        pred_rel_spans = [pred_candidate for i, pred_candidate in enumerate(batch_pred_candidates[idx]) if indices[i] == 1]
        pred_label.append(pred_rel_spans)
        for span_start, span_end in pred_rel_spans:
            assert span_start < seq_len and span_end < seq_len
    return pred_label


def evaluate(data, model, args, name):
    if name == "train":
        instances = data.train_ids
        instances_text = data.train_texts
    elif name == "dev":
        instances = data.dev_ids
        instances_text = data.dev_texts
    elif name == 'test':
        instances = data.test_ids
        instances_text = data.test_texts
    else:
        print("Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []
    model.eval()

    batch_saved_name = args.data_stored_directory + name + f"_all_batch_{args.batch_size}"
    if os.path.exists(batch_saved_name):
        with open(batch_saved_name, 'rb') as fp:
            all_batch = pickle.load(fp)
    else:
        all_batch = construct_all_batch(instances, args.batch_size, args)
        with open(batch_saved_name, 'wb') as fp:
            pickle.dump(all_batch, fp)

    name_to_example_file = {"train": args.train_file, "dev": args.dev_file, "test": args.test_file}
    all_pred_candidates, all_candidates_label, all_pos_tags = construct_candidate_batch(name_to_example_file[name], args, len(all_batch), args.batch_size)

    # model prediction and relation span decode
    for batch_id, batch in enumerate(tqdm(all_batch)):
        batch_pred_candidates = all_pred_candidates[batch_id]
        batch_candidates_label = all_candidates_label[batch_id]
        batch_pos_tags = all_pos_tags[batch_id]
        batch_pred_candidates = [pred_candidates.cuda() for pred_candidates in batch_pred_candidates]
        batch_candidates_label = [candidates_label.cuda() for candidates_label in batch_candidates_label]
        batch_pos_tags = [pos_tags.cuda() for pos_tags in batch_pos_tags]

        batch = tuple(t.cuda() for t in batch)
        char, c_len, pos, gazs, mask, label, recover, t_graph, c_graph, l_graph, adj = batch
        # permutate input !!!
        _, perm_idx = recover.sort(0)
        batch_pred_candidates = permutate_batch(batch_pred_candidates, perm_idx)
        batch_candidates_label = permutate_batch(batch_candidates_label, perm_idx)
        batch_pos_tags = permutate_batch(batch_pos_tags, perm_idx)
        batch_pos_tags = torch.stack(batch_pos_tags, dim=0)

        pred_logits = model(char, c_len, pos, gazs, t_graph, c_graph, l_graph, adj, mask, batch_pred_candidates, batch_pos_tags=batch_pos_tags)
        pred_label = recover_label(pred_logits, batch_pred_candidates, mask, recover)
        pred_results += pred_label
    
    example_predictions = collections.defaultdict(list)
    EvalExample = collections.namedtuple("EvalExample", ["unique_id", "all_predicates"])
    dummy_examples = []  # for predicate_evaluation input
    detailed_predictions = []
    for pred_rel_spans, (chars, _, _, example), instance in zip(pred_results, instances_text, instances):
        unique_id = example["unique_id"]
        sentence = example["sentence"]
        # Note that predicate_evaluate func removes matched elements from all_predicates list, need deepcopy here
        dummy_examples.append(EvalExample(unique_id, example["all_predicates"][:]))
        # gold_predicates = [sentence[char_start:char_end+1]
        #     for char_start, char_end in gold_rel_spans
        # ]
        # assert sorted(gold_predicates) != sorted(example["all_predicates"])
        # assertion fail 60 / 2649
        incomplete_spo_list, incomplete_spo_spans = [], []
        for char_start, char_end in pred_rel_spans:
            example_predictions[unique_id].append((sentence[char_start:char_end+1], char_start, char_end))
            incomplete_spo_list.append(collections.OrderedDict([
                ("subject", ""),
                ("predicate", sentence[char_start:char_end+1]),
                ("object", "")
            ]))
            incomplete_spo_spans.append(collections.OrderedDict([
                ("subject", [-1, -1]),
                ("predicate", [char_start.item(), char_end.item()]),
                ("object", [-1, -1])
            ]))
        detailed_predictions.append(collections.OrderedDict([
            ("unique_id", unique_id),
            ("sentence", sentence),
            ("spo_list", incomplete_spo_list),
            ("spo_spans", incomplete_spo_spans)
        ]))
    
    with open("logs/evaluate/detailed_relation_predictions_dep.json", "w") as writer:
        json.dump(detailed_predictions, writer, ensure_ascii=False, indent=4)
    results = predicate_evaluate(dummy_examples, example_predictions)
    matched_gold = results.pop("matched_gold")

    for key, val in results.items():
        print(f"{name}_{key}: {val}")
    return results


def construct_all_batch(instances, batch_size, args):
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    all_batch = []
    for batch_id in tqdm(range(total_batch)):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch = batchify(instance, gpu=False)
        all_batch.append(batch)
    return all_batch


def get_predicate_candidates(nlp, examples, pos2index):
    Example = collections.namedtuple("Example", ["sentence", "segments"])

    all_pos_tags = []
    all_pred_candidates, all_candidates_label = [], []
    for example in examples:
        sent_spacy = LTP_model_wrapper(nlp, example)
        arg_candidates = make_span_candidates(len(sent_spacy))
        pred_candidates = pred_span_candidates_filter(sent_spacy, arg_candidates, 5)

        segments = [token.text for token in sent_spacy]
        dummy_example = Example(example["sentence"], segments)
        segment_char_span = get_segment_char_span(dummy_example)
        candidates_char_spans = []
        for segment_start, segment_end in pred_candidates:
            char_start = segment_char_span[segment_start][0]
            char_end = segment_char_span[segment_end][1]
            candidates_char_spans.append([char_start, char_end])
        all_pred_candidates.append(candidates_char_spans)

        pos_tags = []
        for token, (char_start, char_end) in zip(sent_spacy, segment_char_span):
            pos_tags += [pos2index[token.tag_]] * (char_end - char_start + 1)
        assert len(pos_tags) == len(example["sentence"])
        all_pos_tags.append(pos_tags)

        gold_char_spans = [spo_span["predicate"] for spo_span in example["spo_spans"]]
        candidates_label = [1 if char_span in gold_char_spans else 0
            for char_span in candidates_char_spans
        ]
        all_candidates_label.append(candidates_label)
    
    # pad pos tags to max length
    max_len = max(len(pos_tags) for pos_tags in all_pos_tags)
    all_padded_pos_tags = [pos_tags + [0] * (max_len - len(pos_tags)) for pos_tags in all_pos_tags]

    # to tensor
    all_pred_candidates = [torch.LongTensor(pred_candidates) for pred_candidates in all_pred_candidates]
    all_candidates_label = [torch.LongTensor(candidates_label) for candidates_label in all_candidates_label]
    all_pos_tags = [torch.LongTensor(pos_tags) for pos_tags in all_padded_pos_tags]
    return all_pred_candidates, all_candidates_label, all_pos_tags


def construct_candidate_batch(file_path, args, n_batch, batch_size):
    # construct predicate candidate batch and label
    with open(file_path, "r") as file:
        raw_train_examples = json.load(file)
    nlp = spacy.load("zh_core_web_md")

    all_pos_tags = []
    all_pred_candidates, all_candidates_label = [], []
    for batch_id in trange(n_batch):
        # produce predicate candidate spans from raw_train_examples
        start_idx, end_idx = batch_id * batch_size, (batch_id + 1) * batch_size
        batch_raw_examples = raw_train_examples[start_idx:end_idx]
        batch_pred_candidates, batch_candidates_label, batch_pos_tags = get_predicate_candidates(nlp, batch_raw_examples, args.pos2index)
        all_pred_candidates.append(batch_pred_candidates)
        all_candidates_label.append(batch_candidates_label)
        all_pos_tags.append(batch_pos_tags)
    return all_pred_candidates, all_candidates_label, all_pos_tags


def permutate_batch(batch_input, perm_idx):
    return [batch_input[idx] for idx in perm_idx]


def train(data, model, args):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "Adam":
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    if args.use_gpu:
        model.cuda()

    batch_saved_name = args.data_stored_directory + f"train_all_batch_{args.batch_size}"
    if os.path.exists(batch_saved_name):
        with open(batch_saved_name, 'rb') as fp:
            all_batch = pickle.load(fp)
    else:
        all_batch = construct_all_batch(data.train_ids, args.batch_size, args)
        with open(batch_saved_name, 'wb') as fp:
            pickle.dump(all_batch, fp)
    all_pred_candidates, all_candidates_label, all_pos_tags = construct_candidate_batch(args.train_file, args, len(all_batch), args.batch_size)

    best_dev = -1
    patience = 30
    num_stuck_epoch = 0
    best_model_names = []
    for idx in range(args.max_epoch):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, args.max_epoch))
        optimizer = lr_decay(optimizer, idx, args.lr_decay, args.lr)
        sample_loss = 0
        total_loss = 0
        # random.shuffle(all_batch)
        model.train()
        model.zero_grad()

        for batch_id, batch in enumerate(tqdm(all_batch)):
            batch_pred_candidates = all_pred_candidates[batch_id]
            batch_candidates_label = all_candidates_label[batch_id]
            batch_pos_tags = all_pos_tags[batch_id]
            batch_pred_candidates = [pred_candidates.cuda() for pred_candidates in batch_pred_candidates]
            batch_candidates_label = [candidates_label.cuda() for candidates_label in batch_candidates_label]
            batch_pos_tags = [pos_tags.cuda() for pos_tags in batch_pos_tags]

            model.zero_grad()
            batch = tuple(t.cuda() for t in batch)
            char, c_len, pos, gazs, mask, label, recover, t_graph, c_graph, l_graph, adj = batch

            # permutate input !!!
            _, perm_idx = recover.sort(0)
            batch_pred_candidates = permutate_batch(batch_pred_candidates, perm_idx)
            batch_candidates_label = permutate_batch(batch_candidates_label, perm_idx)
            batch_pos_tags = permutate_batch(batch_pos_tags, perm_idx)
            batch_pos_tags = torch.stack(batch_pos_tags, dim=0)
            assert char.size(1) == batch_pos_tags.size(1)

            loss = model(char, c_len, pos, gazs, t_graph, c_graph, l_graph, adj, mask,
                         batch_pred_candidates, batch_candidates_label, batch_pos_tags=batch_pos_tags)
            sample_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            if args.use_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            model.zero_grad()
            train_num = len(data.train_ids)
            end = (batch_id+1)*args.batch_size
            if (batch_id + 1) % 100 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f" % (
                        end, temp_cost, sample_loss / 100))
                sys.stdout.flush()
                sample_loss = 0
        
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f" % (end, temp_cost, sample_loss))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss / len(all_batch)))
        # if idx != 3:
        #     continue
        # else:
        #     return
        results = evaluate(data, model, args, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        current_score = results["f1"]
        for key, val in results.items():
            print(f"dev_{key}: {val}")
        if current_score > best_dev:
            print("Exceed previous best metric score:", best_dev)
            if not os.path.exists(args.param_stored_directory + args.dataset_name + "_param"):
                os.makedirs(args.param_stored_directory + args.dataset_name + "_param")
            current_model_name = "{}epoch_{}_metirc_{}.model".format(args.param_stored_directory + args.dataset_name + "_param/", idx, current_score)
            torch.save(model.state_dict(), current_model_name)
            print("Best model saved to {}".format(current_model_name))
            best_model_names.append(current_model_name)
            best_dev = current_score
            num_stuck_epoch = 0

            # only keep the best one
            for model_name in best_model_names[:-1]:
                if os.path.exists(model_name):
                    os.remove(model_name)
                    print("Remove previous best model {}".format(model_name))
        else:
            num_stuck_epoch += 1
            if num_stuck_epoch > patience:
                break
        gc.collect()

    print(f"Loading best model from disk {best_model_names[-1]}")
    model.load_state_dict(torch.load(best_model_names[-1]))


if __name__ == '__main__':
    args, unparsed = get_args()
    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    args.pos_emb_dim = 0
    # args.T = float(unparsed[1])
    # print("Temperature", args.T)
    args.positive_weight = float(unparsed[1])
    print("positive weight", args.positive_weight)
    # seed = args.random_seed
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    data = data_initialization(args)
    # model = PredicateExtractionModel(data, args)
    with open("../Span_OIE/data/pos2index_chinese.json") as f:
        pos2index = json.load(f)
    args.pos2index = {"PAD" : 0}  # [PAD] pos tag -> 0
    args.pos2index.update({pos : index + 1 for pos, index in pos2index.items()})
    args.pos_tag_dim = 20

    model = PredicateSpanScoreModel(data, args)
    # train(data, model, args)

    # model.load_state_dict(torch.load("./logs/distill/3Data_param/epoch_78_metirc_0.621728488043683.model"))
    # model.load_state_dict(torch.load("./logs/debug_relData_param/epoch_57_metirc_0.41418534622134934.model"))
    # model.load_state_dict(torch.load("./logs/debug_rel_spanData_param/epoch_53_metirc_0.502105070424128.model"))  # span pooling
    # model.load_state_dict(torch.load("./logs/debug_rel_spanData_param/epoch_28_metirc_0.4988719429155131.model"))
    # model.load_state_dict(torch.load("./logs/debug_rel_posData_param/epoch_46_metirc_0.530392782233829.model"))  # +pos_tag dep
    # model.load_state_dict(torch.load("./logs/debug_rel_pos_lstmData_param/epoch_24_metirc_0.5153417631476287.model"))  # +pos_tag lstm
    # model.load_state_dict(torch.load("./logs/debug_rel_posData_param/epoch_42_metirc_0.5206101325086155.model"))  # gat layer = 1
    model.load_state_dict(torch.load("./logs/debug_rel_posData_param/epoch_51_metirc_0.5314275242307664.model"))  # gat layer = 3
    # model.load_state_dict(torch.load("./logs/debug_rel_posData_param/epoch_66_metirc_0.535102662910742.model"))  # gat layer = 4
    model.to(device="cuda")
    # evaluate(data, model, args, "train")
    evaluate(data, model, args, "test")
    # evaluate(data, model, args, "dev")
