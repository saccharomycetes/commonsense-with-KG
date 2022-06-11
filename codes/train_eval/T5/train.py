# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import json
import jsonlines

 
logger = logging.getLogger(__name__)

from transformers import MODEL_WITH_LM_HEAD_MAPPING
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_CLASSES = {
	't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer)
}

class MyDataset(torch.utils.data.Dataset):

	def __init__(self, data, pad_token):
		self.data = data
		self.pad_token = pad_token

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx] # one sample
		return sample, self.pad_token

def mCollateFn(batch):
	batch_input_ids = []                                    # len = batch_size * num_cands
	batch_input_masks = []                                  # len = batch_size * num_cands
	batch_input_labels1 = []                                # len = batch_size * num_cands
	batch_input_labels2 = []                                # len = batch_size * num_cands
	in_features = [c for b in batch for c in b[0][0]]       # len = batch_size * num_cands
	label_features1 = [c for b in batch for c in b[0][1]]   # len = batch_size * num_cands
	label_features2 = [c for b in batch for c in b[0][2]]   # len = batch_size * num_cands
	batch_corrects = [b[0][3] for b in batch]               # len = batch_size
	pad_token = batch[0][1]
	max_input_len = max([len(f) for f in in_features])
	max_label1_len = max([len(f) for f in label_features1])
	max_label2_len = max([len(f) for f in label_features2])

	for in_feature, label_feature1, label_feature2 in zip(in_features, label_features1, label_features2):

		in_sequence = in_feature + [pad_token]*(max_input_len-len(in_feature))
		att_mask = [1] * len(in_feature) + [0] * (max_input_len-len(in_feature))
		label_sequence1 = label_feature1 + [pad_token]*(max_label1_len-len(label_features1))
		label_sequence2 = label_feature2 + [pad_token]*(max_label2_len-len(label_features2))
		batch_input_ids.append(in_sequence)
		batch_input_masks.append(att_mask)
		batch_input_labels1.append(label_sequence1)
		batch_input_labels2.append(label_sequence2)
	return batch_input_ids, batch_input_masks, batch_input_labels1, batch_input_labels2, batch_corrects

def mCollateFn_eval(batch):
	batch_input_ids = []                       # len = batch_size * num_cands
	batch_input_masks = []                     # len = batch_size * num_cands
	batch_input_labels = []                    # len = batch_size * num_cands
	batch_corrects = [b[0][3] for b in batch]  # len = batch_size
	batch_dimensions = [b[0][4] for b in batch]
	batch_ids = [b[0][5] for b in batch]

	in_features = [c for b in batch for c in b[0][0]]
	label_features = [c for b in batch for c in b[0][1]]
	pad_token = batch[0][1]
	max_input_len = max([len(f) for f in in_features])
	max_label_len = max([len(f) for f in label_features])

	for in_feature, label_feature in zip(in_features, label_features):

		in_sequence = in_feature + [pad_token]*(max_input_len-len(in_feature))
		att_mask = [1] * len(in_feature) + [0] * (max_input_len-len(in_feature))
		label_sequence = label_feature + [pad_token]*(max_label_len-len(label_feature))
		batch_input_ids.append(in_sequence)
		batch_input_masks.append(att_mask)
		batch_input_labels.append(label_sequence)

	return batch_input_ids, batch_input_masks, batch_input_labels, batch_corrects, batch_dimensions, batch_ids

def convert_examples_to_features(examples, tokenizer):
	data = []
	for example in tqdm(examples, ncols=100):
		if 'dimension' in example: # for CWWV format
			dimension = example['dimension']
			sample_id = str(example['id']) + 'cwwv'
		elif 'dim' in example:   # for ATOMIC format
			dimension = example['dim']
			sample_id = str(example['id']) + 'atomic'
		context = example['context'].replace("___", "<extra_id_0>")
		correct = example["correct"]
		inputs = ["reasoning: " + context + ' ' + sentence for sentence in example["candidates"]]
		num_can = len(inputs)
		input_ids = [tokenizer(ainput, return_tensors='pt').input_ids.numpy().tolist()[0] for ainput in inputs]
		label_ids_r = [tokenizer("1", return_tensors='pt').input_ids.numpy().tolist()[0] for i in range (num_can)]
		label_ids_w = [tokenizer("2", return_tensors='pt').input_ids.numpy().tolist()[0] for i in range (num_can)]
		data.append([input_ids, label_ids_r, label_ids_w, correct, dimension, sample_id])
	return data

def load_and_tokenize(tokenizer, train_file, dev_file):
	with open(train_file)as f_train, open(dev_file)as f_dev:
		train_set = [json.loads(data) for data in f_train.readlines()]
		dev_set = [json.loads(data) for data in f_dev.readlines()]
	train_data = convert_examples_to_features(train_set, tokenizer)
	dev_data = convert_examples_to_features(dev_set, tokenizer)
	return train_data, dev_data


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args, train_dataset, model, tokenizer, eval_dataset):

	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=mCollateFn)
	

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]

	warmup_steps = args.warmup_steps if args.warmup_steps != 0 else int(args.warmup_proportion * t_total)
	logger.info("warm up steps = %d", warmup_steps)
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(0.9, 0.98))
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
	
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
				   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	curr_best = 0.0
	model.zero_grad()
	CE = torch.nn.CrossEntropyLoss(reduction='none')
	loss_fct = torch.nn.MultiMarginLoss(margin=args.margin)
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0], ncols=100)
	epoch = 0
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], ncols=100)
		for step, batch in enumerate(epoch_iterator):
			model.train()

			input_ids = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[0]], dim=0).to(args.device)
			att_mask = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[1]], dim=0).to(args.device)
			input_labels1 = torch.cat([torch.LongTensor(c)[0].view(1,-1) for c in batch[2]], dim=0).to(args.device)
			input_labels2 = torch.cat([torch.LongTensor(c)[0].view(1,-1) for c in batch[3]], dim=0).to(args.device)

			outputs = model(input_ids = input_ids, attention_mask = att_mask, labels = input_labels1)
			logits = outputs[1]
			logits = logits.view(-1,logits.size(-1))
			ce1 = CE(logits, input_labels1.view(-1))
			ce2 = CE(logits, input_labels2.view(-1))
			scores = ce2 - ce1
			scores = scores.view(-1,3)
			corrects = torch.tensor(batch[4], dtype=torch.long)
			corrects = corrects.to(args.device)
			loss = loss_fct(scores, corrects)

			if args.n_gpu > 1:
				loss = loss.mean() # mean() to average on multi-gpu parallel training
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			if args.fp16:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			tr_loss += loss.item()
			
			# one global step stands for one gradient decent
			if (step + 1) % args.gradient_accumulation_steps == 0:
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1
			
				# log step:50
				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
					tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
					tb_writer.add_scalar('Batch_loss', loss.item()*args.gradient_accumulation_steps, global_step)
					logger.info(" global_step = %s, average loss = %s", global_step, (tr_loss - logging_loss)/args.logging_steps)
					logging_loss = tr_loss

				# evalate step
				if args.local_rank == -1 and args.evaluate_during_training and global_step % args.save_steps == 0:
					result = evaluate(args, model, eval_dataset)
					results = {'acc':result}
					for key, value in results.items():
						tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
					if results['acc'] > curr_best:
						curr_best = results['acc']
						# Save model checkpoint
						output_dir = args.output_dir
						if not os.path.exists(output_dir):
							os.makedirs(output_dir)
						model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
						model_to_save.save_pretrained(output_dir)
						tokenizer.save_pretrained(output_dir)
						torch.save(args, os.path.join(output_dir, 'training_args.bin'))
						logger.info("Saving model checkpoint to %s", output_dir)

						# save the best model under checkpoints for each epoch
						epoch_check_dir = os.path.join(args.output_dir,'checkpoint',str(epoch))
						if not os.path.exists(epoch_check_dir):
							os.makedirs(epoch_check_dir)
						model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
						model_to_save.save_pretrained(epoch_check_dir)
						tokenizer.save_pretrained(epoch_check_dir)
						torch.save(args, os.path.join(epoch_check_dir, 'training_args.bin'))

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break

		# get each answer's prob after each epoch
		get_train_dynamics(args, model, tokenizer, train_dataset, epoch)
		# do it later
		epoch+=1

	result = evaluate(args, model, eval_dataset)
	results = {'acc':result}
	for key, value in results.items():
		tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
	if results['acc'] > curr_best:
		curr_best = results['acc']
		# Save model checkpoint
		output_dir = args.output_dir
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
		model_to_save.save_pretrained(output_dir)
		tokenizer.save_pretrained(output_dir)
		torch.save(args, os.path.join(output_dir, 'training_args.bin'))
		logger.info("Saving model checkpoint to %s", output_dir)
	if args.local_rank in [-1, 0]:
		tb_writer.close()
	return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset):
	right_num = 0
	total_num = 0
	if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
		os.makedirs(args.output_dir)

	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=mCollateFn_eval)

	# Eval!
	logger.info("***** Running evaluation *****")
	logger.info("  Num examples = %d", len(eval_dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)

	for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=100):
		model.eval()
		with torch.no_grad():
			input_ids = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[0]], dim=0).to(args.device)
			att_mask = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[1]], dim=0).to(args.device)
			input_labels = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[2]], dim=0).to(args.device)
			logits = model(input_ids = input_ids, attention_mask = att_mask, labels = input_labels)[1]
			logits = logits.view(logits.size(0),-1)
			scores = logits[:,209] - logits[:,204]
			scores = scores.view(-1,3)
			answers = torch.argmax(scores, dim=1)
			corrects = torch.LongTensor(batch[3]).to(args.device)

			right_num += int(torch.sum(answers.eq(corrects)))
			total_num += len(batch[3])
	output_eval_file = os.path.join(args.output_dir, args.results_file)
	with open(output_eval_file, "w") as writer:
		logger.info("***** Eval results *****")
		logger.info("  acc = %s", str(right_num/total_num))
		writer.write("acc = %s\n" % (str(right_num/total_num)))
	return right_num/total_num

def get_train_dynamics(args, model, tokenizer, train_dataset, epoch):

	results = {}
	if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
		os.makedirs(args.output_dir)
	
	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
	eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=mCollateFn_eval)

	# Eval!
	preds = []
	out_label_ids = []
	dimensions = []
	sample_ids = []
	correct_number = []
	for batch in tqdm(eval_dataloader, desc="Mean Cal", ncols=100):
		model.eval()
		with torch.no_grad():
			input_ids = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[0]], dim=0).to(args.device)
			att_mask = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[1]], dim=0).to(args.device)
			input_labels = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[2]], dim=0).to(args.device)
			logits = model(input_ids = input_ids, attention_mask = att_mask, labels = input_labels)[1]
			logits = logits.view(logits.size(0),-1)
			scores = logits[:,209] - logits[:,204]
			scores = scores.view(-1,3)
		preds.append(scores)
		out_label_ids.append(np.array(batch[3]))
		# out_label_ids example: [array([0, 2, 2, 1, 1, 2, 0, 1])]
		dimensions.append(batch[4])
		# example dimensions: [['taxonomic', 'taxonomic', 'lexical', 'taxonomic', 
		# 'similarity', 'rel-other', 'similarity', 'taxonomic']]
		sample_ids.append(batch[5])

	preds = torch.cat(preds, dim=0).cpu().numpy()  # (样本数，3)
	#  preds[0] :     array([ -7.8149376,  -6.4938107, -10.834098 ], dtype=float32)
	pred_label_ids = np.argmax(preds, axis=1)             # (样本数,)    预测label
	# pred_label_ids[0]:  1
	out_label_ids = np.concatenate(out_label_ids, axis=0) #（样本数,） 真实label
	# out_label_ids[0]: 0
	dimensions = [sample for batch in dimensions for sample in batch] # list len(dimensions) = 样本数
	sample_ids = [sample for batch in sample_ids for sample in batch]

	def softmax(x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	stat_folder = os.path.join(args.output_dir,'stats')
	if not os.path.exists(stat_folder):
		os.makedirs(stat_folder)

	file_name = f"{epoch}_mean.tsv"
	output_stat_file = os.path.join(stat_folder,file_name)
	with open(output_stat_file,'w') as f:
		for index, each_sample in enumerate(preds):
			probs  = softmax(each_sample)
			gold_label_index = out_label_ids[index] # gold label 	
			pred_label_index = pred_label_ids[index] # predict label    
			prob = probs[gold_label_index]
			dimension = dimensions[index]
			sample_id = sample_ids[index]
			str_probs = ','.join([str(i) for i in probs])

			f.write(f"{sample_id}\t{dimension}\t")
			for i in probs:
				f.write(f"{i}\t")
			f.write(f"{gold_label_index}\t{pred_label_index}\n")

			# f.write(f"{sample_id}\t{dimension}\t{str_probs}\t{gold_label_index}\t{pred_label_index}\n")

			# if gold_label_index == pred_label_index:
			# 	f.write(f"{sample_id}\t{dimension}\t{prob}\t1\n")
			# else:
			# 	f.write(f"{sample_id}\t{dimension}\t{prob}\t0\n")


def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--train_file", default=None, type=str, required=True,
						help="The train file name")
	parser.add_argument("--dev_file", default=None, type=str, required=True,
						help="The dev file name")
	parser.add_argument("--model_type", default=None, type=str, required=True,
						help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
	parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
						help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_TYPES))
	parser.add_argument("--config_name", default="", type=str,
						help="Pretrained config name or path if not the same as model_name")
	parser.add_argument("--tokenizer_name", default="", type=str,
						help="Pretrained tokenizer name or path if not the same as model_name")
	parser.add_argument("--cache_dir", default="", type=str,
						help="Where do you want to store the pre-trained models downloaded from s3")
	parser.add_argument("--task_name", default=None, type=str, required=True,
						help="The name of the task to train selected in the list: ")
	parser.add_argument("--output_dir", default=None, type=str, required=True,
						help="The output directory where the model predictions and checkpoints will be written.")

	## Other parameters
	parser.add_argument("--second_train_file", default=None, type=str,
						help="Used when combining ATOMIC and CWWV")
	parser.add_argument("--second_dev_file", default=None, type=str,
						help="Used when combining ATOMIC and CWWV")
	parser.add_argument("--max_seq_length", default=128, type=int,
						help="The maximum total input sequence length after tokenization. Sequences longer "
							 "than this will be truncated, sequences shorter will be padded.")
	parser.add_argument("--max_words_to_mask", default=6, type=int,
						help="The maximum number of tokens to mask when computing scores")
	parser.add_argument("--max_sequence_per_time", default=80, type=int,
						help="The maximum number of sequences to feed into the model")
	parser.add_argument("--do_train", action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval", action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--evaluate_during_training", action='store_true',
						help="Run evaluation during training at each logging step.")
	parser.add_argument("--do_lower_case", action='store_true',
						help="Set this flag if you are using an uncased model.")
	parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
						help="Batch size per GPU/CPU for training.")
	parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
						help="Batch size per GPU/CPU for evaluation.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--margin", default=1.0, type=float,
						help="The margin for ranking loss")
	parser.add_argument("--learning_rate", default=1e-5, type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.01, type=float,
						help="Weight deay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-6, type=float,
						help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")
	parser.add_argument("--num_train_epochs", default=1.0, type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--max_steps", default=-1, type=int,
						help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
	parser.add_argument("--warmup_steps", default=0, type=int,
						help="Linear warmup over warmup_steps.")
	parser.add_argument("--warmup_proportion", default=0.05, type=float,
						help="Linear warmup over warmup proportion.")
	parser.add_argument('--logging_steps', type=int, default=50,
						help="Log every X updates steps.")
	parser.add_argument('--save_steps', type=int, default=50,
						help="Save checkpoint every X updates steps.")
	parser.add_argument("--logits_file", default='logits_test.txt', type=str, 
						help="The file where prediction logits will be written")
	parser.add_argument("--results_file", default='eval_results.txt', type=str,
						help="The file where eval results will be written")
	parser.add_argument("--no_cuda", action='store_true',
						help="Avoid using CUDA when available")
	parser.add_argument('--overwrite_output_dir', action='store_true',
						help="Overwrite the content of the output directory")
	parser.add_argument('--seed', type=int, default=2555,
						help="random seed for initialization")
	parser.add_argument('--fp16', action='store_true',
						help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
	parser.add_argument('--fp16_opt_level', type=str, default='O1',
						help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
							 "See details at https://nvidia.github.io/apex/amp.html")
	parser.add_argument("--local_rank", type=int, default=-1,
						help="For distributed training: local_rank")
	parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
	parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")


	args = parser.parse_args()

	if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir and args.do_train:
		raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend='nccl')
		args.n_gpu = 1
	args.device = device

	if args.do_train:
		for handler in logging.root.handlers[:]:
			logging.root.removeHandler(handler)
	# Setup logging
	if args.do_train:
		log_file = os.path.join(args.output_dir, 'train.log')
		logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
							datefmt = '%m/%d/%Y %H:%M:%S',
							level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
							filename=log_file)
		logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
						args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
		os.system("cp train.py %s" % os.path.join(args.output_dir, 'train_s10.py'))

	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, finetuning_task=args.task_name, cache_dir=args.cache_dir)
	tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
	model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir)

	count = count_parameters(model)
	print ("parameters:"+str(count))

	if args.local_rank == 0:
		torch.distributed.barrier()

	model.to(args.device)

	logger.info("Training/evaluation parameters %s", args)

	train_data, dev_data = load_and_tokenize(tokenizer, args.train_file, args.dev_file)

	eval_dataset = MyDataset(dev_data, tokenizer.pad_token_id)

	if args.do_train:
		train_dataset = MyDataset(train_data, tokenizer.pad_token_id)
		global_step, tr_loss = train(args, train_dataset, model, tokenizer, eval_dataset)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
	results = {}
	if args.do_eval:
		tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
		model = model_class.from_pretrained(args.output_dir)
		model.eval()
		model.to(args.device)
		result = evaluate(args, model, eval_dataset)
	return results

if __name__ == "__main__":
	main()