# commonsense-with-KG
This repository contains the code for the paper "An Empirical Investigation of Commonsense Self-Supervision with Knowledge Graphs" see full paper [here](https://arxiv.org/abs/2205.10661)
## `data_process` folder:
The "make_quatiles.py" is used to generate the piqa data quartiles based on 3 terms.

The "x%.py" is used to generate the random x% of a training data.(which we used for data_size analysis)

The "T5_eval_maker.py '' is used to transform the raw 5 benchmarks into T5 forms data.

The train&eval folder contains the train & evalation code for the 4 model sets mentioned in the paper: Roberta and T5 for J loss and I loss.
​
To run the data process codes:
```
python make_quatiles.py \
--piqa_data_dir {your piqa data} \
--out_dir {your output directory} \
--task {which term you use to rank the data, should be one of: sim, len, overlap} \
--vocab_file {your train file if you are computing vocab_overlap}
​
python x%.py \
--ratio {the ratio of choose data} \
--data_i {input data} \
--data_o {output data}
​
python T5_eval_maker.py \
--dev_data_dir {you dev data directory} \
--out_dir {output directory}
```
## `train&eval` folder
It contains the train&eval code for the 4 model sets mentioned in the paper: Roberta and T5 for J loss and I loss.
T5 train and evaluate:
```
CUDA_VISIBLE_DEVICES={} python train.py \
--train_file {train file} \
--dev_file {dev file} \
--model_type t5 \
--model_name_or_path {which model you use} \
--task_name cskg \
--output_dir {output directory} \
--cache_dir ../pre_model \
--per_gpu_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-5 --num_train_epochs 5 --warmup_proportion 0.05 \
--evaluate_during_training --per_gpu_eval_batch_size 8  --save_steps 6500 \
--do_train
​
CUDA_VISIBLE_DEVICES={}} python eval.py \
--model_dir {model directory} \
--eval_dir {evaluate data directory} \
--eval_batch_size 4
```

Roberta train and evaluate:
​
```
CUDA_VISIBLE_DEVICES={} python train.py \
--model_type roberta-mlm \
--model_name_or_path {the model you use} \
--task_name atomic \
--output_dir {output directory} \
--max_sequence_per_time 200 \
--train_file {train file} \
--dev_file {dev file} \
--cache_dir ../pre_model \
--max_seq_length 128 \
--max_words_to_mask 6 \
--do_train --do_eval \
--per_gpu_train_batch_size 2  \
--gradient_accumulation_steps 16 \
--learning_rate 1e-5 --num_train_epochs 5 --warmup_proportion 0.05 --evaluate_during_training \
--per_gpu_eval_batch_size 8  --save_steps 6500 --margin 1.0

python evaluate_RoBERTa.py \
--lm  {model directory} \
--dataset_file {evaluate data directory} \
--out_dir {output directory} \
--device {the GPU you use} \
--reader {the bench mark you are evaluating}
```

## Our data:
You can find our data [here](https://drive.google.com/drive/folders/12rPpe7vbkxfIDTSSYYJaCmO1nfD8eHF6?usp=sharing)

It contains our 100% training data, you can use the "x%.py" to make it into random x% of our data.

It also contains data for development (evaluate during training) and 5 benchmarks' evaluate data, you can use the "T5_eval_maker.py" to transform it into a T5 evaluating format (which is not needed for training).

You can use the "make_quatiles.py" to generate the piqa data quartiles based on 3 terms, it's worth noting that if you want to evaluate the terms using T5, you should replace the eval.py line 63-69 with the annotated ones.
