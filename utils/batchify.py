import torch
from utils.graph_generator import *


def batchify(input_batch_list, gpu):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    gazs = [sent[1] for sent in input_batch_list]
    labels = [sent[2] for sent in input_batch_list]
    inst_type = "entity" if input_batch_list[0][3] is not None else "relation"
    if len(input_batch_list[0]) > 3:
        pos_list = [sent[3] for sent in input_batch_list]

    word_seq_lengths = list(map(len, words))
    max_seq_len = max(word_seq_lengths)
    gazs_list, gaz_lens, max_gaz_len = seq_gaz(gazs)
    # TODO: t c l graph self-loop
    tmp_matrix = list(map(graph_generator, [(max_gaz_len, max_seq_len, gaz) for gaz in gazs]))
    batch_t_matrix = torch.ByteTensor([ele[0] for ele in tmp_matrix])
    batch_c_matrix = torch.ByteTensor([ele[1] for ele in tmp_matrix])
    batch_l_matrix = torch.ByteTensor([ele[2] for ele in tmp_matrix])
    gazs_tensor = torch.zeros((batch_size, max_gaz_len), requires_grad=False).long()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=False).byte()
    pos_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    if inst_type == "entity":
        label_seq_tensor = torch.zeros((batch_size, 4), requires_grad=False).long()
    else:
        label_seq_tensor = torch.zeros((batch_size, max_seq_len, 2), requires_grad=False).long()

    for idx, (seq, pos, gaz, gaz_len, label, seqlen) in enumerate(zip(words, pos_list, gazs_list, gaz_lens, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        gazs_tensor[idx, :gaz_len] = torch.LongTensor(gaz)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        if inst_type == "entity":
            pos_seq_tensor[idx, :seqlen] = torch.LongTensor(pos)
            label_seq_tensor[idx, :] = torch.LongTensor([*label[0], *label[1]])
        else:
            # for relation instance, pos_seq_tensor is all-zero
            label_seq_tensor[idx, :seqlen, :] = torch.LongTensor(label).transpose(0, 1)

    word_seq_lengths = torch.LongTensor(word_seq_lengths)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    pos_seq_tensor = pos_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    gazs_tensor = gazs_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    batch_t_matrix = batch_t_matrix[word_perm_idx]
    batch_c_matrix = batch_c_matrix[word_perm_idx]
    batch_l_matrix = batch_l_matrix[word_perm_idx]

    if len(input_batch_list[0]) > 4:
        segment_char_spans = [sent[4] for sent in input_batch_list]
        dependency_heads = [sent[5] for sent in input_batch_list]
        dependency_rels = [sent[6] for sent in input_batch_list]

        max_segment_len = max([len(segment_char_span) for segment_char_span in segment_char_spans])
        adj_matrix = [dependency_graph_generator(segment_char_span, dependency_head, max_seq_len, max_segment_len)
                        for segment_char_span, dependency_head in zip(segment_char_spans, dependency_heads)]
        adj_matrix = torch.FloatTensor(adj_matrix)
        batch_adj_matrix = adj_matrix[word_perm_idx]

    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        pos_seq_tensor = pos_seq_tensor.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
        batch_t_matrix = batch_t_matrix.cuda()
        gazs_tensor = gazs_tensor.cuda()
        batch_c_matrix = batch_c_matrix.cuda()
        batch_l_matrix = batch_l_matrix.cuda()
    return word_seq_tensor, word_seq_lengths, pos_seq_tensor, gazs_tensor, mask, label_seq_tensor, word_seq_recover, batch_t_matrix, batch_c_matrix, batch_l_matrix, batch_adj_matrix



