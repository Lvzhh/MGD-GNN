from train_argument import *
import argparse


def merge_linguistic_info(gold_file, rel_pred_file, output_file):
    """ Merge linguistic info in gold_file to relation_prediction file. """
    print(f"Merge dependency info in {gold_file} to {rel_pred_file}, write result to {output_file}.")
    # merge
    with open(rel_pred_file, "r") as file:
        relation_predictions = json.load(file)
    with open(gold_file, "r") as file:
        test_data = json.load(file)

    id_to_orig_example = dict(
        (orig_example["unique_id"], orig_example)
        for orig_example in test_data
    )
    for example in relation_predictions:
        unique_id = example["unique_id"]
        orig_example = id_to_orig_example[unique_id]
        example["ltp_segments"] = orig_example["ltp_segments"]
        example["ltp_pos"] = orig_example["ltp_pos"]
        example["ltp_dependency"] = orig_example["ltp_dependency"]
    unique_ids = set(example["unique_id"] for example in relation_predictions)
    all_unique_ids = set(example["unique_id"] for example in test_data)
    assert unique_ids == all_unique_ids
    with open(output_file, "w") as writer:
        writer.write(json.dumps(relation_predictions, ensure_ascii=False, indent=4) + "\n")


def predict(data, model, args):
    instances = data.test_ids
    instances_text = data.test_texts

    pred_results = []
    gold_results = []
    model.eval()
    batch_size = args.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1

    for batch_id in tqdm(range(total_batch)):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch = batchify(instance, gpu=False)
        batch = tuple(t.cuda() for t in batch)
        char, c_len, pos, gazs, mask, label, recover, t_graph, c_graph, l_graph, dep_graph = batch
        pred_logits = model(char, c_len, pos, gazs, t_graph, c_graph, l_graph, dep_graph, mask)
        pred_label, gold_label = recover_label(pred_logits, label, mask, data.label_alphabet, recover)
        pred_results += pred_label
        gold_results += gold_label
    
    orig_example_triples = collections.defaultdict(list)
    orig_example_predictions = collections.defaultdict(list)
    for pred, gold, (chars, _, _, example), instance in zip(pred_results, gold_results, instances_text, instances):
        subj_text = example["sentence"][pred[0] : pred[1] + 1]
        obj_text = example["sentence"][pred[2] : pred[3] + 1]
        orig_id = example["orig_id"]
        # assert example["subject"] == example["sentence"][gold[0] : gold[1] + 1]
        # assert example["object"] == example["sentence"][gold[2] : gold[3] + 1]
        orig_example_triples[orig_id].append([example["subject"], example["predicate"], example["object"]])
        orig_example_predictions[orig_id].append([subj_text, example["predicate"], obj_text])
    
    results = entity_evaluate(orig_example_triples, orig_example_predictions)
    orig_example_pred_result = results.pop("example_pred_result")
    for key, val in results.items():
        print(f"test_{key}: {val}")

    # write predictions with their sentences (s, p, o)
    detailed_predictions = []
    processed_examples = set()
    for (chars, _, _, example) in instances_text:
        orig_id = example["orig_id"]
        if orig_id in processed_examples:
            continue
        example_with_prediction = collections.OrderedDict(
            [
                ("unique_id", orig_id),
                ("sentence", example["sentence"]),
                ("gold", orig_example_triples[orig_id]),
                ("predictions", orig_example_predictions[orig_id]),
                ("pred_result", orig_example_pred_result[orig_id])
            ]
        )
        detailed_predictions.append(example_with_prediction)
        processed_examples.add(orig_id)

    output_detailed_predictions_file = os.path.join("./logs/evaluate", "detailed_argument_predictions.json")
    with open(output_detailed_predictions_file, "w") as writer:
        writer.write(json.dumps(detailed_predictions, ensure_ascii=False, indent=4) + "\n")


if __name__ == '__main__':
    args, unparsed = get_args()
    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    seed = args.random_seed

    eval_parser = argparse.ArgumentParser()
    eval_parser.add_argument('--gold_file', type=str, default="./data/test.json")
    eval_parser.add_argument('--rel_pred_file', type=str, default="/data/lzh/exp/OpenIE/chinese-openie/logs/rerun/joint/detailed_relation_predictions.json")
    eval_args = eval_parser.parse_args(unparsed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    data = data_initialization(args)
    # Note that max_sentence_length = 250 would ignore some examples in prediction phrase
    # data.max_sentence_length = 300

    merge_output_file = "./logs/evaluate/detailed_predicate_predictions_with_linguistic_info.json"
    print("data.max_sentence_length", data.max_sentence_length)
    merge_linguistic_info(eval_args.gold_file, eval_args.rel_pred_file, merge_output_file)
    data.generate_instance(merge_output_file, "test")
    print("test data cut_num", data.test_cut_num)
    model = BLSTM_GAT_CRF(data, args)
    # train(data, model, args)
    
    # model.load_state_dict(torch.load("./logs/split_check/lstm_1Data_param/epoch_21_metirc_0.5550743739408774.model"))  # lstm
    # model.load_state_dict(torch.load("./logs/split_check/dep_1Data_param/epoch_28_metirc_0.6023347768781774.model"))  # dep
    # model.load_state_dict(torch.load("./logs/debug_depData_param/epoch_28_metirc_0.5578987008096403.model"))  # gat layer = 1
    # model.load_state_dict(torch.load("./logs/debug_depData_param/epoch_46_metirc_0.6106194690265486.model"))  # gat layer = 3
    # model.load_state_dict(torch.load("./logs/debug_depData_param/epoch_70_metirc_0.6179627188853323.model"))  # gat layer = 4
    # model.load_state_dict(torch.load("./logs/distill/3Data_param/epoch_75_metirc_0.6207870457540953.model"))
    model.load_state_dict(torch.load("./logs/argumentData_param/epoch_31_metirc_0.5923554886085372.model"))
    model.to(device="cuda")
    predict(data, model, args)
    # evaluate(data, model, args, "test")
    # evaluate(data, model, args, "train")
