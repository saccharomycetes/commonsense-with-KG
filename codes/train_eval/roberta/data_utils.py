import csv
import logging
from tqdm import tqdm
import json
import re
import ftfy
import random
from collections import Counter
import unicodedata
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
skip_words = set(stopwords.words('english'))
skip_words.add('\'s')
skip_words.add('.')
skip_words.add(',')
PERSON_NAMES = ['Alex', 'Ash', 'Aspen', 'Bali', 'Berkeley', 'Cameron', 'Chris', 'Cody', 'Dana', 'Drew', 'Emory', 'Flynn', 'Gale', 'Jamie', 'Jesse', 
'Kai', 'Kendall', 'Kyle', 'Lee', 'Logan', 'Max', 'Morgan', 'Nico', 'Paris', 'Pat', 'Quinn', 'Ray', 'Robin', 'Rowan', 'Rudy', 'Sam', 'Skylar', 'Sydney', 
'Taylor', 'Tracy', 'West', 'Wynne']
logger = logging.getLogger(__name__)

def accuracy(out, labels):
	return {'acc': (out == labels).mean()}

def handle_words(span, tokenizer, keywords=None, is_start=False):
	inputs = []
	labels = []
	words = nltk.word_tokenize(span)
	for w_i, w in enumerate(words):
		if (w_i == 0 and is_start) or w == '.' or w == ',' or w.startswith('\''):
			w_bpes = tokenizer.tokenize(w)
		else:
			w_bpes = tokenizer.tokenize(w, add_prefix_space=True)
		inputs.extend(w_bpes)
		if keywords != None:
			if w in keywords:
				labels.extend(w_bpes)
			else:
				labels.extend([-100]*len(w_bpes))
		else:
			if w not in PERSON_NAMES and w not in skip_words and w.lower() not in skip_words:
				labels.extend(w_bpes)
			else:
				labels.extend([-100]*len(w_bpes))
	return inputs, labels

def handle_underscores(suffix, tokenizer, keywords=None, prefix=False):
	inputs = []
	labels = []
	if '_' in suffix:
		suffix_parts = [i.strip() for i in suffix.split('___')]
		for i, part in enumerate(suffix_parts):
			if part:
				tmp_inputs, tmp_labels = handle_words(part, tokenizer, keywords=keywords, is_start=(i==0 and prefix))
				inputs += tmp_inputs
				labels += tmp_labels

				if i != len(suffix_parts) - 1 and suffix_parts[i+1]:
					inputs.append(tokenizer.mask_token)
					labels.append(-100) # -100 => skip words?
			else:
				inputs.append(tokenizer.mask_token)
				labels.append(-100)
	else:
		inputs, labels = handle_words(suffix, tokenizer, keywords=keywords, is_start=prefix)
	return inputs, labels

#将一条知识转化成三句话，两种形式
def convert_examples_to_features(examples, tokenizer, max_length=512):
	data = []
	for example in tqdm(examples, desc="DATA"):
		inputs, labels = handle_underscores(example['context'], tokenizer, keywords=example['keywords'], prefix=True)
		# e.g.  inputs: ['aching', 'Ġis', 'Ġsimilar', 'Ġto']
		# e.g.  labels: ['aching', -100, -100, -100]

		# here cannot understand the difference between input_ids and label_ids ?????
		choices = [handle_underscores(cand, tokenizer) for cand in example['candidates']]
		##  choices example:
		## [(['Ġpainful', '.'], ['Ġpainful', -100]), 
		## (['Ġun', 'sh', 'aven', '.'], ['Ġun', 'sh', 'aven', -100]), 
		## (['Ġhigh', 'Ġpotential', '.'], ['Ġhigh', 'Ġpotential', -100])]
		
		input_ids = [inputs+cand[0] for cand in choices]
		input_ids = [tokenizer.convert_tokens_to_ids(cand) for cand in input_ids]
		label_ids = [labels+cand[1] for cand in choices]
		label_ids = [[t if t == -100 else input_ids[i][t_i] for t_i, t in enumerate(cand)] for i, cand in enumerate(label_ids)]
		label_ids = [[-100]+cand+[-100] for cand in label_ids]
		input_ids = [tokenizer.prepare_for_model(cand, max_length=max_length, truncation=True)['input_ids'] for cand in input_ids]
		
		### input_ids  => question + answer 
		### label_ids  => question + answer - skip words.

		## input_ids 
		# [[0, 13341, 16, 1122, 7, 8661,             4, 2], 
		## [0, 13341, 16, 1122, 7, 542, 1193, 10570, 4, 2], 
		## [0, 13341, 16, 1122, 7, 239, 801,         4, 2]]

		## label_ids 
		##[[-100, 13341, -100, -100, -100,   8661,                -100, -100], 
		## [-100, 13341, -100, -100, -100,   542, 1193, 10570,    -100, -100], 
		## [-100, 13341, -100, -100, -100,   239, 801,            -100, -100]]

		if 'dimension' in example: # for CWWV format
			dimension = example['dimension']
			sample_id = str(example['id']) + 'cwwv'	
			data.append([input_ids, label_ids, example['correct'],dimension,sample_id])	
		elif'dim' in example:  # for ATOMIC format
			dimension = example['dim']
			sample_id = str(example['id']) + 'atomic'	
			data.append([input_ids, label_ids, example['correct'],dimension,sample_id])	
		else:
			# ignore this sample
			pass
				
	return data

class ATOMICProcessor(object):
	def __init__(self, args):
		print ('loading from %s %s' % (args.train_file, args.dev_file))
		self.filelist = [args.train_file, args.dev_file]
		self.D = [[], []]

	def get_train_examples(self):
		self.load_data(self.filelist[0], 0)
		return self.D[0]

	def get_dev_examples(self):
		self.load_data(self.filelist[1], 1)
		return self.D[1]

	def load_data(self, filename, sid):
		with open(filename, "r") as f:
			for row in tqdm(f):
				sample = json.loads(row) # load all attributes
				# e.g.
				# {"id": "358787", "dim": "xReact", "context": "Sam gets a sunburn. As a result, Sam felt", "correct": 2, 
				# "candidates": ["like looser.", "contented.", "upset."], "keywords": ["gets", "sunburn"]}
				self.D[sid].append(sample)
			print (len(self.D[sid]))

class CWWVProcessor(object):
	def __init__(self, args):
		self.answerKey_mapping = {'A':0, 'B':1, 'C':2}
		self.D = [[], []]
		if args.task_name == 'cskg':
			print ('loading from %s %s' % (args.second_train_file, args.second_dev_file))
			self.filelist = [args.second_train_file, args.second_dev_file]
		else:
			print ('loading from %s %s' % (args.train_file, args.dev_file))
			self.filelist = [args.train_file, args.dev_file]

	def get_train_examples(self):
		self.load_data(self.filelist[0], 0)
		return self.D[0]

	def get_dev_examples(self):
		self.load_data(self.filelist[1], 1)
		return self.D[1]

	def load_data(self, filename, sid):
		skipped = 0
		with open(filename, "r") as f:
			for row in tqdm(f):
				sample = json.loads(row)
				context = sample['question']['stem']
				if context.endswith('.'):
					context = context[:-1]
				if not context.endswith('[MASK]'):
					skipped += 1
					context_parts = context.split('[MASK]')
					context = context_parts[0].strip()
					candidates = [c['text']+context_parts[1]+'.' for c in sample['question']['choices']]
				else:
					context = context[:-7]
					candidates = [c['text']+'.' for c in sample['question']['choices']]
				label = self.answerKey_mapping[sample['answerKey']]
				keywords = nltk.word_tokenize(sample['question']['head'])
				keywords = [w for w in keywords if w not in skip_words and w.lower() not in skip_words]
				dimension = sample['question']['dimension'] # add dimension property
				self.D[sid].append({'id':sample['id'], 'context':context, 'correct':label, 'candidates':candidates, 'keywords':keywords,'dimension':dimension})
			print(len(self.D[sid]), skipped)

class CSKGProcessor(object):
	def __init__(self, args):
		# CWWV set always uses second train/dev file params 
		self.atomicprocessor = ATOMICProcessor(args)
		self.cwwvprocessor = CWWVProcessor(args)

	def get_train_examples(self):
		cwwv_questions = self.cwwvprocessor.get_train_examples()
		atomic_questions = self.atomicprocessor.get_train_examples()
		return cwwv_questions+atomic_questions

	def get_dev_examples(self):
		cwwv_questions = self.cwwvprocessor.get_dev_examples()
		atomic_questions = self.atomicprocessor.get_dev_examples()
		return cwwv_questions+atomic_questions

myprocessors = {
	"atomic": ATOMICProcessor,
	"cwwv": CWWVProcessor,
	"cskg": CSKGProcessor
}
