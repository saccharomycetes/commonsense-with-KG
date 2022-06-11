import argparse
import os
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import json

class MyDataset(torch.utils.data.Dataset):

	def __init__(self, data, pad_token):
		self.data = data
		self.pad_token = pad_token

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx] # one sample
		return sample, self.pad_token

def load_and_tokenize(tokenizer, dev_file):
	with open(dev_file)as f_dev:
		dev_set = [json.loads(data) for data in f_dev.readlines()]
	dev_data = convert_examples_to_features(dev_set, tokenizer)
	return dev_data

def mCollateFn(batch):
	batch_input_ids = []                       # len = batch_size * num_cands
	batch_input_masks = []                     # len = batch_size * num_cands
	batch_labels = []                          # len = batch_size * num_cands
	batch_corrects = [b[0][2] for b in batch]  # len = batch_size

	in_features = [i for b in batch for i in b[0][0]]
	label_features = [i for b in batch for i in b[0][1]]
	pad_token = batch[0][1]
	max_input_len = max([len(f) for f in in_features])
	max_label_len = max([len(f) for f in label_features])
	
	for in_feature, label_feature in zip(in_features, label_features):

		in_sequence = in_feature + [pad_token] * (max_input_len-len(in_feature))
		att_mask = [1] * len(in_feature) + [0] * (max_input_len-len(in_feature))
		label_sequence = label_feature + [pad_token]*(max_label_len-len(label_feature))

		batch_input_ids.append(in_sequence)
		batch_input_masks.append(att_mask)
		batch_labels.append(label_sequence)
	return batch_input_ids, batch_input_masks, batch_labels, batch_corrects

def convert_examples_to_features(examples, tokenizer):
	data = []
	# this part is for quartiles analysising
	# for example in examples:
	# 	contexts = ["reasoning: " + example["goal"] + " " + example["sol1"], "reasoning: " + example["goal"] + " " + example["sol2"]]
	# 	correct = example["label"]
	# 	candidates = ["1", "1"]
	# 	input_ids = [tokenizer(context, return_tensors='pt').input_ids.numpy().tolist()[0] for context in contexts]
	# 	label_id = [tokenizer(candidate, return_tensors='pt').input_ids.numpy().tolist()[0] for candidate in candidates]
	# 	data.append([input_ids, label_id, correct])
	for example in examples:
		contexts = example["context"]
		correct = example["correct"]
		candidates = example["candidates"]
		input_ids = [tokenizer(context, return_tensors='pt').input_ids.numpy().tolist()[0] for context in contexts]
		label_id = [tokenizer(candidate, return_tensors='pt').input_ids.numpy().tolist()[0] for candidate in candidates]
		data.append([input_ids, label_id, correct])
	return data

def evaluate(args, model, eval_dataset):
	right_num = 0
	total_num = 0

	# Note that DistributedSampler samples randomly
	CE = torch.nn.CrossEntropyLoss(reduction='none')
	eval_sampler = SequentialSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=mCollateFn)

	# Eval!
	num_can = 0
	for batch in tqdm(eval_dataloader, desc="Evaluating", ncols = 100):
		model.eval()
		with torch.no_grad():
			input_ids = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[0]], dim=0).to(args.device)
			att_mask = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[1]], dim=0).to(args.device)
			input_labels = torch.cat([torch.LongTensor(c)[0].view(1,-1) for c in batch[2]], dim=0).to(args.device)
			outputs = model(input_ids = input_ids, attention_mask = att_mask, labels = input_labels)
			if num_can == 0:
				num_can = int(input_ids.size(0)/args.eval_batch_size)
			logits = outputs[1]
			logits = logits.view(-1,logits.size(-1))
			scores = logits[:,209] - logits[:,204] # only use the prob of "1" and "2"
			scores = scores.view(-1,num_can)
			answers = torch.argmax(scores, dim=1)
			corrects = torch.LongTensor(batch[3]).to(args.device)
			right_num += int(torch.sum(answers.eq(corrects)))
			total_num += len(batch[3])
	print(right_num/total_num)
	acc = right_num/total_num
	if not os.path.exists(args.out_dir + "/" + args.model_dir.split("/")[-1]):
		os.mkdir(args.out_dir + "/" + args.model_dir.split("/")[-1])
	out_dir = args.out_dir + "/" + args.model_dir.split("/")[-1]
	output_eval_file = out_dir + "/" + args.eval_dir.split("/")[-1][:4] + "_acc.txt"
	with open(output_eval_file, "w") as writer:
		writer.write(f"Accuracy: {acc:.3f}")
	return right_num/total_num

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, type=str, required=True,
					    help="The model dir for eval")
    parser.add_argument("--eval_dir", default=None, type=str, required=True,
					    help="The dir of data for eval")
    parser.add_argument("--eval_batch_size", default=1, type=int,
						help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--out_dir", default=None, type=str,
						help="Out directory for the accuracy")			
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.eval_dir.split("/")[-1][:4])
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir, do_lower_case=True)
    dev_data = load_and_tokenize(tokenizer, args.eval_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.eval()
    model.to(args.device)
    eval_dataset = MyDataset(dev_data, tokenizer.pad_token_id)
    result = evaluate(args, model, eval_dataset)
    return result

if __name__ == "__main__":
	main()