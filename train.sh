
# train predicate extraction model
CUDA_VISIBLE_DEVICES=6 python train_predicate_span.py --train_file data/train.json --dev_file data/dev.json --test_file data/test.json --gat_nhead 5 --gat_layer 3 --strategy n --batch_size 50 --lr 0.001 --lr_decay 0.01 --use_clip False --optimizer Adam --droplstm 0 --dropout 0.6 --dropgat 0.3 --gaz_dropout 0.4 --norm_char_emb True --norm_gaz_emb False --param_stored_directory ./logs/predicate --lstm_layer 1 --gat_nhidden 60 --data_stored_directory ./logs/generated_data_predicate/ --positive_weight 3


# train argument extraction model
CUDA_VISIBLE_DEVICES=5 python train_argument.py --train_file data/train.json --dev_file data/dev.json --test_file data/test.json --gat_nhead 5 --gat_layer 3 --strategy n --batch_size 50 --lr 0.001 --lr_decay 0.01 --use_clip False --optimizer Adam --droplstm 0 --dropout 0.6 --dropgat 0.3 --gaz_dropout 0.4 --norm_char_emb True --norm_gaz_emb False --param_stored_directory ./logs/argument --lstm_layer 1 --gat_nhidden 60 --data_stored_directory ./logs/generated_data_argument/


# predict
# change line 132 to best argument model path, then run the following cmds
CUDA_VISIBLE_DEVICES=6 python predict_argument.py --gold_file data/test.json --rel_pred_file logs/evaluate/detailed_predicate_predictions.json --gat_nhead 5 --gat_layer 2 --strategy n --batch_size 50 --norm_char_emb True --norm_gaz_emb False --param_stored_directory ./logs/argument --lstm_layer 1 --gat_nhidden 60 --data_stored_directory ./logs/generated_data_argument/

python utils/eval_joint.py --gold_file data/test.json --pred_file ./logs/evaluate/detailed_argument_predictions.json --output_dir ./logs/evaluate