# Multi-Grained Dependency Graph Neural Network for Chinese Open Information Extraction

Source code for PAKDD 2021 paper [Multi-Grained Dependency Graph Neural Network for Chinese Open Information Extraction](https://link.springer.com/chapter/10.1007/978-3-030-75768-7_13).

## Requirements

- Python version >= 3.6
- [PyTorch](http://pytorch.org/) version >= 1.2.0


## Running

### Pre-steps

Download character embeddings gigaword_chn.all.a2b.uni.ite50.vec ([Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)) and word embeddings sgns.merge.word ([Google Drive](https://drive.google.com/file/d/1Zh9ZCEu8_eSQ-qkYVQufQDNKPC4mtEKR/view) or [Baidu Pan](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw)).

Change utils/config.py line 34 and 35 to your word and character embedding file path.

### Training

Run the following command to train a predicate extraction model:

```
CUDA_VISIBLE_DEVICES=0 python train_predicate_span.py --train_file data/train.json --dev_file data/dev.json --test_file data/test.json --gat_nhead 5 --gat_layer 3 --strategy n --batch_size 50 --lr 0.001 --lr_decay 0.01 --use_clip False --optimizer Adam --droplstm 0 --dropout 0.6 --dropgat 0.3 --gaz_dropout 0.4 --norm_char_emb True --norm_gaz_emb False --param_stored_directory ./logs/predicate --lstm_layer 1 --gat_nhidden 60 --data_stored_directory ./logs/generated_data_predicate/ --positive_weight 3
```

To train an argument extraction model, run the following command:

```
CUDA_VISIBLE_DEVICES=0 python train_argument.py --train_file data/train.json --dev_file data/dev.json --test_file data/test.json --gat_nhead 5 --gat_layer 3 --strategy n --batch_size 50 --lr 0.001 --lr_decay 0.01 --use_clip False --optimizer Adam --droplstm 0 --dropout 0.6 --dropgat 0.3 --gaz_dropout 0.4 --norm_char_emb True --norm_gaz_emb False --param_stored_directory ./logs/argument --lstm_layer 1 --gat_nhidden 60 --data_stored_directory ./logs/generated_data_argument/
```

### Evaluation

Change predict_argument.py line 132 to best argument model path, then run the following commands:

``` 
CUDA_VISIBLE_DEVICES=0 python predict_argument.py --gold_file data/test.json --rel_pred_file logs/evaluate/detailed_predicate_predictions.json --gat_nhead 5 --gat_layer 2 --strategy n --batch_size 50 --norm_char_emb True --norm_gaz_emb False --param_stored_directory ./logs/argument --lstm_layer 1 --gat_nhidden 60 --data_stored_directory ./logs/generated_data_argument/

python utils/eval_joint.py --gold_file data/test.json --pred_file ./logs/evaluate/detailed_argument_predictions.json --output_dir ./logs/evaluate
```


## Citation

If the code helps you, please cite our paper:

```bibtex
@InProceedings{lyu2021MGDGNN,
	author="Lyu, Zhiheng
	and Shi, Kaijie
	and Li, Xin
	and Hou, Lei
	and Li, Juanzi
	and Song, Binheng",
	title="Multi-Grained Dependency Graph Neural Network for Chinese Open Information Extraction",
	booktitle="Advances in Knowledge Discovery and Data Mining",
	year="2021",
	publisher="Springer International Publishing",
	address="Cham",
	pages="155--167",
}
```

This repo is adapted from [Graph4CNER](https://github.com/DianboWork/Graph4CNER). We thank them for their work.
