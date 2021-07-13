import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
# from layer.crf import CRF
from layer.gatlayer import GAT, GCN, HeteGAT
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BLSTM_GAT_CRF(nn.Module):
    def __init__(self, data, args):
        super(BLSTM_GAT_CRF, self).__init__()
        print("build BLSTM_GAT_CRF model...")
        self.name = "BLSTM_GAT_CRF"
        self.strategy = args.strategy
        self.char_emb_dim = data.char_emb_dim
        self.gaz_emb_dim = data.gaz_emb_dim
        self.pos_emb_dim = args.pos_emb_dim  # position embedding
        self.pos_tag_dim = args.pos_tag_dim if hasattr(args, "pos_tag_dim") else 0  # part-of-speech dim
        self.gaz_embeddings = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.char_embeddings = nn.Embedding(data.char_alphabet.size(), self.char_emb_dim)
        self.pos_tag_embeddings = nn.Embedding(len(args.pos2index) if hasattr(args, "pos2index") else 0, self.pos_tag_dim)
        # self.pos_tag_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(len(args.pos2index), self.pos_tag_dim)))

        if self.pos_emb_dim > 0:
            self.pos_embeddings = nn.Embedding(args.max_sentence_length * 2, self.pos_emb_dim)
            self.pos_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(args.max_sentence_length * 2, self.pos_emb_dim)))
                
        if data.pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.char_alphabet.size(), self.char_emb_dim)))
        if data.pretrain_gaz_embedding is not None:
            self.gaz_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)))
        if args.fix_gaz_emb:
            self.gaz_embeddings.weight.requires_grad = False
        else:
            self.gaz_embeddings.weight.requires_grad = True

        self.hidden_dim = self.gaz_emb_dim
        self.bilstm_flag = args.bilstm_flag
        self.lstm_layer = args.lstm_layer
        if self.bilstm_flag:
            lstm_hidden = self.hidden_dim // 2
        else:
            lstm_hidden = self.hidden_dim
        self.lstm = nn.LSTM(self.char_emb_dim + self.pos_emb_dim + self.pos_tag_dim, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        # self.gcn = GCN(self.hidden_dim, self.hidden_dim, 0, args.dropgat, 2)
        # self.hidden2hidden = nn.Linear(self.hidden_dim, crf_input_dim)

        crf_input_dim = 4
        # self.gat_1 = GAT(self.hidden_dim, args.gat_nhidden, crf_input_dim, args.dropgat, args.alpha, args.gat_nhead, args.gat_layer)
        # self.gat_2 = GAT(self.hidden_dim, args.gat_nhidden, crf_input_dim, args.dropgat, args.alpha, args.gat_nhead, args.gat_layer)
        self.gat_3 = GAT(self.hidden_dim, args.gat_nhidden, crf_input_dim, args.dropgat, args.alpha, args.gat_nhead, args.gat_layer)
        # self.hetegat = HeteGAT(self.hidden_dim, args.gat_nhidden, crf_input_dim, args.dropgat, args.alpha, args.gat_nhead, args.gat_layer)
        # if self.strategy == "v":
        #     self.weight1 = nn.Parameter(torch.ones(crf_input_dim))
        #     self.weight2 = nn.Parameter(torch.ones(crf_input_dim))
        #     self.weight3 = nn.Parameter(torch.ones(crf_input_dim))
        #     self.weight4 = nn.Parameter(torch.ones(crf_input_dim))
        # elif self.strategy == "n":
        #     self.weight1 = nn.Parameter(torch.ones(1))
        #     self.weight2 = nn.Parameter(torch.ones(1))
        #     self.weight3 = nn.Parameter(torch.ones(1))
        #     self.weight4 = nn.Parameter(torch.ones(1))
        # else:
        #     self.weight = nn.Linear(crf_input_dim*4, crf_input_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.droplstm = nn.Dropout(args.droplstm)
        self.gaz_dropout = nn.Dropout(args.gaz_dropout)
        self.reset_parameters()

        self.subject_outputs = nn.Linear(self.hidden_dim, 2)
        self.object_outputs = nn.Linear(self.hidden_dim, 2)

    def reset_parameters(self):
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0_reverse)
        nn.init.orthogonal_(self.lstm.weight_ih_l0_reverse)
        # nn.init.orthogonal_(self.hidden2hidden.weight)
        # nn.init.constant_(self.hidden2hidden.bias, 0)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def _get_lstm_features(self, batch_char, batch_len, batch_pos=None, batch_pos_tags=None):
        embeds = self.char_embeddings(batch_char)
        if self.pos_emb_dim > 0:
            pos_embeds = self.pos_embeddings(batch_pos)
            embeds = torch.cat([embeds, pos_embeds], dim=-1)
        if self.pos_tag_dim > 0:
            pos_tag_embeds = self.pos_tag_embeddings(batch_pos_tags)
            embeds = torch.cat([embeds, pos_tag_embeds], dim=-1)
        embeds = self.dropout(embeds)
        embeds_pack = pack_padded_sequence(embeds, batch_len, batch_first=True)
        out_packed, (_, _) = self.lstm(embeds_pack)
        lstm_feature, _ = pad_packed_sequence(out_packed, batch_first=True)
        lstm_feature = self.droplstm(lstm_feature)
        return lstm_feature

    def _get_gcn_feature(self, batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph):
        gaz_feature = self.gaz_embeddings(gaz_list)
        gaz_feature = self.gaz_dropout(gaz_feature)
        lstm_feature = self._get_lstm_features(batch_char, batch_len, batch_pos)
        max_seq_len = lstm_feature.size()[1]
        gat_input = torch.cat((lstm_feature, gaz_feature), dim=1)
        # gcn_feature = self.gcn(gat_input, l_graph.to(dtype=torch.float32))
        # gcn_feature = gcn_feature[:, :max_seq_len, :]

        # gat_feature_1 = self.gat_1(gat_input, t_graph)
        # gat_feature_1 = gat_feature_1[:, :max_seq_len, :]
        # gat_feature_2 = self.gat_2(gat_input, c_graph)
        # gat_feature_2 = gat_feature_2[:, :max_seq_len, :]
        gat_feature_3 = self.gat_3(gat_input, l_graph)
        gat_feature_3 = gat_feature_3[:, :max_seq_len, :]
        gcn_feature = gat_feature_3
        # lstm_feature = self.hidden2hidden(lstm_feature)
        # if self.strategy == "m":
        #     gcn_feature = torch.cat((lstm_feature, gat_feature_1, gat_feature_2, gat_feature_3), dim=2)
        #     gcn_feature = self.weight(gcn_feature)
        # elif self.strategy == "v":
        #     gcn_feature = torch.mul(lstm_feature, self.weight1) + torch.mul(gat_feature_1, self.weight2) + torch.mul(
        #         gat_feature_2, self.weight3) + torch.mul(gat_feature_3, self.weight4)
        # else:
        #     gcn_feature = self.weight1 * lstm_feature + self.weight2 * gat_feature_1 + self.weight3 * gat_feature_2 + self.weight4 * gat_feature_3
        return gcn_feature

    def _get_dep_feature(self, batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph, adj, batch_pos_tags=None):
        lstm_feature = self._get_lstm_features(batch_char, batch_len, batch_pos, batch_pos_tags=batch_pos_tags)
        max_seq_len = lstm_feature.size()[1]
        segment_char_adj = adj[:, max_seq_len:, :max_seq_len]
        segment_hidden = (segment_char_adj @ lstm_feature) / (torch.sum(segment_char_adj, dim=-1, keepdim=True) + 1e-8)
        # [batch_size, max_segment_length, hidden_size]
        gat_input = torch.cat((lstm_feature, segment_hidden), dim=1)

        gat_feature_3 = self.gat_3(gat_input, adj)
        gat_feature_3 = gat_feature_3[:, :max_seq_len, :]
        gcn_feature = gat_feature_3
        return gcn_feature
    
    def _get_depgaz_feature(self, batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph, dep_graph):
        lstm_feature = self._get_lstm_features(batch_char, batch_len, batch_pos)
        max_seq_len = lstm_feature.size()[1]
        segment_char_adj = dep_graph[:, max_seq_len:, :max_seq_len]
        segment_hidden = (segment_char_adj @ lstm_feature) / (torch.sum(segment_char_adj, dim=-1, keepdim=True) + 1e-8)
        max_segment_len = segment_char_adj.size(1)
        # [batch_size, max_segment_len, hidden_size]
        gaz_feature = self.gaz_embeddings(gaz_list)
        gaz_feature = self.gaz_dropout(gaz_feature)
        gat_input = torch.cat((lstm_feature, segment_hidden, gaz_feature), dim=1)
        max_gaz_len = gaz_feature.size(1)
        adj = torch.eye(max_seq_len + max_segment_len + max_gaz_len).to(dep_graph)
        adj = adj.unsqueeze(0).expand(dep_graph.size(0), -1, -1)
        seq_segment_len = max_seq_len + max_segment_len
        adj[:, :seq_segment_len, :seq_segment_len] = dep_graph
        # TODO: add sentence seq edge
        # adj[:, :max_seq_len, :max_seq_len] = l_graph[:, :max_seq_len, :max_seq_len]
        adj[:, :max_seq_len, seq_segment_len:] = l_graph[:, :max_seq_len, max_seq_len:]
        adj[:, seq_segment_len:, :max_seq_len] = l_graph[:, max_seq_len:, :max_seq_len]

        # gat_feature_3 = self.gat_3(gat_input, adj)
        gat_feature_3 = self.hetegat(gat_input, adj, max_seq_len, max_segment_len, max_gaz_len)
        gat_feature_3 = gat_feature_3[:, :max_seq_len, :]
        gcn_feature = gat_feature_3
        return gcn_feature

    def forward(self, batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph, dep_graph, mask, batch_label=None, batch_teacher_logits=None, T=1.0):
        # gcn_feature = self._get_lstm_features(batch_char, batch_len, batch_pos)
        # gcn_feature = self._get_gcn_feature(batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph)
        gcn_feature = self._get_dep_feature(batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph, dep_graph)
        # gcn_feature = self._get_depgaz_feature(batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph, dep_graph)
        
        sequence_output = gcn_feature
        sub_logits = self.subject_outputs(sequence_output)
        ob_logits = self.object_outputs(sequence_output)

        if batch_label is not None:
            sub_start_logits, sub_end_logits = sub_logits.split(1, dim=-1)
            ob_start_logits, ob_end_logits = ob_logits.split(1, dim=-1)
            
            sub_start_logits = sub_start_logits.squeeze(-1).masked_fill(mask == 0, -1e9)
            sub_end_logits = sub_end_logits.squeeze(-1).masked_fill(mask == 0, -1e9)
            ob_start_logits = ob_start_logits.squeeze(-1).masked_fill(mask == 0, -1e9)
            ob_end_logits = ob_end_logits.squeeze(-1).masked_fill(mask == 0, -1e9)

            sub_start_pos, sub_end_pos, ob_start_pos, ob_end_pos = [part_label.squeeze(-1) for part_label in batch_label.split(1, dim=-1)]

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = sub_start_logits.size(1)
            sub_start_pos.clamp_(0, ignored_index)
            sub_end_pos.clamp_(0, ignored_index)
            ob_start_pos.clamp_(0, ignored_index)
            ob_end_pos.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            sub_start_loss = loss_fct(sub_start_logits, sub_start_pos)
            sub_end_loss = loss_fct(sub_end_logits, sub_end_pos)
            ob_start_loss = loss_fct(ob_start_logits, ob_start_pos)
            ob_end_loss = loss_fct(ob_end_logits, ob_end_pos)
            total_loss = (sub_start_loss + sub_end_loss + ob_start_loss + ob_end_loss) / 4

            if batch_teacher_logits is not None:
                kl_div_loss = nn.KLDivLoss(reduction="batchmean")
                teacher_loss = 0
                T = 3.0
                for student_logits, teacher_logits in zip([sub_start_logits, sub_end_logits, ob_start_logits, ob_end_logits],
                                                          batch_teacher_logits):
                    teacher_loss += kl_div_loss(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1))
                    assert torch.sum((student_logits == -1e9) != (teacher_logits == -1e9)) == 0
                #     print(student_logits, torch.max(student_logits))
                #     print(teacher_logits, torch.max(teacher_logits))
                #     print(teacher_loss)
                # exit(0)
                total_loss += teacher_loss  # TODO weight KD loss
            return total_loss
        else:
            predictions = torch.cat([sub_logits, ob_logits], dim=-1)
            return predictions


class PredicateExtractionModel(nn.Module):
    def __init__(self, data, args):
        super(PredicateExtractionModel, self).__init__()
        self.encoder = BLSTM_GAT_CRF(data, args)
        self.hidden_dim = data.gaz_emb_dim
        self.predicate_outputs = nn.Linear(self.hidden_dim, 2)
        # TODO init predicate_outputs
    
    def forward(self, batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph, dep_graph, mask, batch_label=None):
        # gcn_feature = self.encoder._get_lstm_features(batch_char, batch_len, batch_pos)
        gcn_feature = self.encoder._get_dep_feature(batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph, dep_graph)
        sequence_output = gcn_feature

        pred_logits = self.predicate_outputs(sequence_output)
        pred_start_logits, pred_end_logits = pred_logits.split(1, dim=-1)
        pred_start_logits = pred_start_logits.squeeze(-1)
        pred_end_logits = pred_end_logits.squeeze(-1)

        if batch_label is not None:
            pred_start_positions, pred_end_positions = [t.squeeze(-1).to(dtype=torch.float32) for t in batch_label.split(1, dim=-1)]
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            start_loss = criterion(pred_start_logits, pred_start_positions)
            end_loss = criterion(pred_end_logits, pred_end_positions)
            mask = mask.to(dtype=torch.float32)
            len_from_mask = torch.sum(mask, dim=-1).to(dtype=torch.long)
            assert torch.sum(batch_len != len_from_mask) == 0

            start_loss = torch.sum(start_loss * mask) / torch.sum(mask)
            end_loss = torch.sum(end_loss * mask) / torch.sum(mask)
            total_loss = start_loss + end_loss
            return total_loss
        else:
            return pred_logits


class PredicateSpanScoreModel(nn.Module):
    def __init__(self, data, args):
        super(PredicateSpanScoreModel, self).__init__()
        self.encoder = BLSTM_GAT_CRF(data, args)
        self.hidden_dim = data.gaz_emb_dim
        self.span_classifier = nn.Linear(self.hidden_dim * 4, 2)
        self.positive_weight = args.positive_weight
        # TODO init predicate_outputs

    def forward(self, batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph, dep_graph, mask, batch_span_candidates, batch_label=None, batch_pos_tags=None):
        # gcn_feature = self.encoder._get_lstm_features(batch_char, batch_len, batch_pos, batch_pos_tags=batch_pos_tags)
        gcn_feature = self.encoder._get_dep_feature(batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph, dep_graph, batch_pos_tags=batch_pos_tags)
        batch_sequence_output = gcn_feature
        # [batch_size, max_seq_len, hidden_size]

        batch_span_logits = []
        for idx, (sequence_output, span_candidates) in enumerate(zip(batch_sequence_output, batch_span_candidates)):
            # get span features
            # span_features = []
            # for span_start, span_end in span_candidates:
            #     assert span_start < batch_len[idx] and span_end < batch_len[idx], (batch_len[idx], span_start, span_end)
            #     start_feature = sequence_output[span_start]
            #     end_feature = sequence_output[span_end]
            #     # TODO add dep span feature
            #     span_feature = torch.cat([start_feature, end_feature, start_feature + end_feature, start_feature - end_feature], dim=0)
            #     span_features.append(span_feature)
            # span_features = torch.stack(span_features, dim=0)  # [n_candidates, hidden_dim * 4]

            # get span features
            # span_candidates: [n_candidates, 2]
            span_starts, span_ends = span_candidates[:, 0], span_candidates[:, 1]
            start_features = sequence_output[span_starts]
            end_features = sequence_output[span_ends]
            # span_poolings = []
            # for start, end in zip(span_starts, span_ends):
            #     span_output = sequence_output[start:end + 1]
            #     span_poolings.append(torch.cat([torch.mean(span_output, dim=0), torch.max(span_output, dim=0).values]))
            # span_poolings = torch.stack(span_poolings, dim=0)
            # span_features = torch.cat([start_features, end_features, span_poolings], dim=-1)
            span_features = torch.cat([start_features, end_features, start_features + end_features, start_features - end_features], dim=-1)
            # [n_candidates, hidden_dim * 4]

            span_logits = self.span_classifier(span_features)
            batch_span_logits.append(span_logits)

        if batch_label is not None:
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1, self.positive_weight], dtype=torch.float32).cuda())
            loss = 0
            for span_logits, label in zip(batch_span_logits, batch_label):
                loss += loss_fct(span_logits, label)
            return loss
        else:
            return batch_span_logits


class PredicateWordSpanScoreModel(nn.Module):
    def __init__(self, data, args):
        super(PredicateWordSpanScoreModel, self).__init__()
        self.encoder = BLSTM_GAT_CRF(data, args)
        self.hidden_dim = data.gaz_emb_dim
        self.span_classifier = nn.Linear(self.hidden_dim * 4, 2)
        self.positive_weight = args.positive_weight
        # TODO init predicate_outputs

    def forward(self, batch_char, batch_len, batch_pos, gaz_list, t_graph, c_graph, l_graph, dep_graph, mask, batch_span_candidates, batch_label=None, batch_pos_tags=None):
        # gcn_feature = self.encoder._get_lstm_features(batch_char, batch_len, batch_pos, batch_pos_tags=batch_pos_tags)
        adj = dep_graph
        lstm_feature = self.encoder._get_lstm_features(batch_char, batch_len, batch_pos, batch_pos_tags=batch_pos_tags)
        max_seq_len = lstm_feature.size()[1]
        segment_char_adj = adj[:, max_seq_len:, :max_seq_len]
        segment_hidden = (segment_char_adj @ lstm_feature) / (torch.sum(segment_char_adj, dim=-1, keepdim=True) + 1e-8)
        # [batch_size, max_segment_length, hidden_size]
        gat_input = torch.cat((lstm_feature, segment_hidden), dim=1)

        adj[:, max_seq_len:, :max_seq_len] = 0
        adj[:, :max_seq_len, max_seq_len:] = 0
        gat_feature_3 = self.encoder.gat_3(gat_input, adj)
        gat_feature_3 = gat_feature_3[:, max_seq_len:, :]
        gcn_feature = gat_feature_3

        batch_sequence_output = gcn_feature
        # [batch_size, max_seq_len, hidden_size]

        batch_span_logits = []
        for idx, (sequence_output, span_candidates) in enumerate(zip(batch_sequence_output, batch_span_candidates)):
            # get span features
            # span_candidates: [n_candidates, 2]
            span_starts, span_ends = span_candidates[:, 0], span_candidates[:, 1]
            start_features = sequence_output[span_starts]
            end_features = sequence_output[span_ends]
            span_features = torch.cat([start_features, end_features, start_features + end_features, start_features - end_features], dim=-1)
            # [n_candidates, hidden_dim * 4]

            span_logits = self.span_classifier(span_features)
            batch_span_logits.append(span_logits)

        if batch_label is not None:
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1, self.positive_weight], dtype=torch.float32).cuda())
            loss = 0
            for span_logits, label in zip(batch_span_logits, batch_label):
                loss += loss_fct(span_logits, label)
            return loss
        else:
            return batch_span_logits
