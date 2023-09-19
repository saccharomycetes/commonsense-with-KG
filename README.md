# commonsense-with-KG
This repository contains the code for the paper "A Study of Zero-shot Adaptation with Commonsense Knowledge", 
See the full paper [here](https://www.akbc.ws/2022/assets/pdfs/3_a_study_of_zero_shot_adaptatio.pdf).

The models developed in the paper have shown promising results on multiple commonsense benchmarks, narrowing the gap with supervised models. Moreover, our efforts to capture key dependencies between the selected knowledge, the language model, and the properties of the task make the approach highly adaptable to new benchmarks and applications.

## Code for gerneral evaluation on our model

In our work, we have used the syhthetic data constructed from knowledge graph to enhance the language models. Here, we provide the best model's weights and evluation codes, please note that all the evaluation codes are in a multople-choice manner.

### Intuition of the result given by our model

Our models were enhanced by [CSKG](https://arxiv.org/pdf/2012.11490.pdf), so the result given by our models shows how much does knowledge graph (in our way) help the language model to better perform on the testing dataset. For example, commonsense knowledge graph may be useful for guiding language models answering Physics questions (as is shown in our paper).

The link to our model weights is [here](https://drive.google.com/drive/folders/1EA-3iRWePo_u9FtOt-C6D9ZtV6gCLfBT?usp=sharing).

Firstly, you should clone this repository, and download the model weights to the "models" folder.

```
git clone https://github.com/saccharomycetes/commonsense-with-KG.git
cd commonsense-with-KG
```

And create an environment and install the required packages:

```
conda create -n cskglm python=3.8
conda activate cskglm
pip install -r requirements.txt
```

You should also download the model weights to the "models" folder.

To run the general evluation code on your dataset, firstly you should organize your dataset to a json file, which is a list of list, each list contains all of the candidates, the model will pick the one that is most align with its learned commonsense knowledge.

An example evluation code is shown below, where we use the [PIQA](https://arxiv.org/pdf/1906.05433.pdf) dataset as an example, the piqa dataset is ready to use in the "codes/general_eval", you can run the following code to test the result:

```
python codes/general_eval/general_eval.py \
--lm models/roberta_large \
--dataset_file codes/general_eval/piqa.json \
--out_dir codes/general_eval \
--device 0
```

To test on your own dataset, transfer the dataset into the same format as "piqa.json", and change the "dataset_file" to your dataset file path, change the "out_dir" to your desired output directory, and change the "device" to the GPU you want to use.


# Code for paper reproduction

## `data_process` folder:
The "make_quatiles.py" is used to generate the piqa data quartiles based on 3 terms.

The "x%.py" is used to generate the random x% of a training data. (which we used for data_size analysis)

The "T5_eval_maker.py '' is used to transform the raw 5 benchmarks into T5 forms data.

The train&eval folder contains the train & evalation code for the 2 model sets mentioned in the paper: Roberta and T5.
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
It contains the train&eval code for the 2 model sets mentioned in the paper: Roberta and T5.
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


# Cite 

If you find our dataset or code to be useful in your research, please consider citing the following paper:

```
@article{zhang2022study,
  title={A Study of Zero-shot Adaptation with Commonsense Knowledge},
  author={Zhang, Jiarui and Ilievski, Filip and Ma, Kaixin and Francis, Jonathan and Oltramari, Alessandro},
  journal={Automated Knowledge Base Construction (AKBC)},
  year={2022}
}
```

## Contact

-   `jrzhang [AT] isi.edu`