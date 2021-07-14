import sys

from utils.data import Data
from utils.batchify import batchify
from utils.config import get_args
from model.bilstm_gat_crf import BLSTM_GAT_CRF
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
from tqdm import tqdm
from utils.metric import entity_evaluate


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
        data.generate_instance(args.train_file, "train", False)
        data.generate_instance(args.dev_file, "dev")
        data.generate_instance(args.test_file, "test")
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


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _get_best_span(start_logits, end_logits, seq_len):
    start_indexes = _get_best_indexes(start_logits, 20)
    end_indexes = _get_best_indexes(end_logits, 20)
    pred_span = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            if seq_len <= start_index or seq_len <= end_index or \
                end_index < start_index:
                continue
            pred_span.append([start_index, end_index, start_logits[start_index] + end_logits[end_index]])
    if len(pred_span) == 0:
        pred_span.append([0, -1, 0])
    best_span = sorted(pred_span, key=lambda x: x[2], reverse=True)[0]
    return best_span[:2]


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, 4): pred tag result
            gold_variable (batch_size, 4): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_logits = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        seq_logits = pred_logits[idx]
        seq_len = np.sum(mask[idx])
        subj_span = _get_best_span(seq_logits[:, 0], seq_logits[:, 1], seq_len)
        obj_span = _get_best_span(seq_logits[:, 2], seq_logits[:, 3], seq_len)
        pred_label.append([*subj_span, *obj_span])
        gold_label.append(gold_tag[idx].tolist())
    return pred_label, gold_label


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

    batch_saved_name = args.data_stored_directory + name + "_all_batch"
    if os.path.exists(batch_saved_name):
        with open(batch_saved_name, 'rb') as fp:
            all_batch = pickle.load(fp)
    else:
        all_batch = construct_all_batch(instances, args.batch_size, args)
        with open(batch_saved_name, 'wb') as fp:
            pickle.dump(all_batch, fp)

    for batch_id, batch in enumerate(tqdm(all_batch)):
        batch = tuple(t.cuda() for t in batch)
        char, c_len, pos, gazs, mask, label, recover, t_graph, c_graph, l_graph, adj = batch
        pred_logits = model(char, c_len, pos, gazs, t_graph, c_graph, l_graph, adj, mask)
        pred_label, gold_label = recover_label(pred_logits, label, mask, data.label_alphabet, recover)
        pred_results += pred_label
        gold_results += gold_label
    
    orig_example_triples = collections.defaultdict(list)
    orig_example_predictions = collections.defaultdict(list)
    for pred, gold, (chars, _, _, example), instance in zip(pred_results, gold_results, instances_text, instances):
        subj_text = example["sentence"][pred[0] : pred[1] + 1]
        obj_text = example["sentence"][pred[2] : pred[3] + 1]
        orig_id = example["orig_id"]
        assert example["subject"] == example["sentence"][gold[0] : gold[1] + 1]
        assert example["object"] == example["sentence"][gold[2] : gold[3] + 1]
        orig_example_triples[orig_id].append([example["subject"], example["predicate"], example["object"]])
        orig_example_predictions[orig_id].append([subj_text, example["predicate"], obj_text])

    results = entity_evaluate(orig_example_triples, orig_example_predictions)
    orig_example_pred_result = results.pop("example_pred_result")

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


def train(data, model, args):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "Adam":
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    if args.use_gpu:
        model.cuda()

    batch_saved_name = args.data_stored_directory + "train_all_batch"
    if os.path.exists(batch_saved_name):
        with open(batch_saved_name, 'rb') as fp:
            all_batch = pickle.load(fp)
    else:
        all_batch = construct_all_batch(data.train_ids, args.batch_size, args)
        with open(batch_saved_name, 'wb') as fp:
            pickle.dump(all_batch, fp)

    # # construct teacher logits batch
    # all_teacher_batch = []
    # teacher_logits_path = "../chinese-openie/logs/rerun/distill/eval_logits.pt"
    # all_teacher_logits = torch.load(teacher_logits_path)
    # batch_size = args.batch_size
    # for batch_id, batch in enumerate(tqdm(all_batch)):
    #     sent_chars, sent_lens, recover = batch[0], batch[1], batch[6]
    #     seq_len = sent_chars.size()[1]
    #     start = batch_id * batch_size
    #     end = min(start + batch_size, len(all_teacher_logits))
    #     batch_teacher_logits = all_teacher_logits[start:end]
    #     if not batch_teacher_logits:
    #         continue
        
    #     sent_lens = sent_lens[recover]
    #     teacher_logits_tensor = torch.full((end - start, 4, seq_len), -1e9, dtype=torch.float32)
    #     for idx, sent_logits in enumerate(batch_teacher_logits):
    #         sent_len = sent_logits.size(1)
    #         assert sent_len == sent_lens[idx]
    #         teacher_logits_tensor[idx][:, :sent_len] = sent_logits
    #     # permute
    #     _, perm_index = recover.sort(0)
    #     rev_recover = torch.zeros(perm_index.size()[0], dtype=torch.long)
    #     for idx, pos in enumerate(recover):
    #         rev_recover[pos.item()] = idx
    #     assert all(perm_index == rev_recover)
    #     teacher_logits_tensor = teacher_logits_tensor[perm_index]
    #     all_teacher_batch.append([t.squeeze(1).cuda() for t in teacher_logits_tensor.split(1, dim=1)])

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
            model.zero_grad()
            batch = tuple(t.cuda() for t in batch)
            char, c_len, pos, gazs, mask, label, recover, t_graph, c_graph, l_graph, adj = batch

            # loss = model(char, c_len, pos, gazs, t_graph, c_graph, l_graph, adj, mask, label, all_teacher_batch[batch_id], T=args.T)
            loss = model(char, c_len, pos, gazs, t_graph, c_graph, l_graph, adj, mask, label, T=args.T)
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
        # if idx != args.max_epoch - 1:
        #     continue
        # else:
        #     return
        results = evaluate(data, model, args, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        current_score = results["all_accuracy"]
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

    model.load_state_dict(torch.load(best_model_names[-1]))


if __name__ == '__main__':
    args, unparsed = get_args()
    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    args.T = 3.0
    # args.T = float(unparsed[1])
    # print("Temperature", args.T)
    # seed = args.random_seed
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    data = data_initialization(args)
    model = BLSTM_GAT_CRF(data, args)
    train(data, model, args)
    # model.load_state_dict(torch.load("./logs/distill/3Data_param/epoch_78_metirc_0.621728488043683.model"))
    # model.to(device="cuda")
    evaluate(data, model, args, "test")
    # evaluate(data, model, args, "dev")
