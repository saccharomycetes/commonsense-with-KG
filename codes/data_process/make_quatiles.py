# this code is used to generate te piqa data quatiles based on 3 terms.

import json
import argparse
from tqdm import tqdm
from transformers import RobertaTokenizerFast


def rank_quatiles(task, in_dir, tokenizer, vocab_file=None):
    if task == "overlap":
        with open(vocab_file)as f:
            qas = f.readlines()
            vocabs = {}
            for qa in tqdm(qas, ncols=70):
                example = json.loads(qa)
                context = example['context']
                can = example['candidates']
                for text in [context] + can:
                    tokens = tokenizer(text)['input_ids']
                    for token in tokens:
                        if token in vocabs:
                            vocabs[token] += 1
                        else:
                            vocabs[token] = 1
    with open(in_dir)as f:
        qas = f.readlines()
        examples = []
        for qa in tqdm(qas, ncols=70):
            example = json.loads(qa)
            if task == "sim":
                words1 = set(tokenizer(example['sol1'])['input_ids'])
                words2 = set(tokenizer(example['sol2'])['input_ids'])
                uni = words1.union(words2)
                inte = words1.intersection(words2)
                jaccard = len(inte)/len(uni)
                this_example = {k:v for k,v in example.items()}
                this_example['term'] = jaccard
            elif task == "len":
                can1 = example['goal'] + " " + example["sol1"]
                can2 = example['goal'] + " " + example["sol2"]
                len1 = len(tokenizer(can1)['input_ids'])
                len2 = len(tokenizer(can2)['input_ids'])
                this_example = {k:v for k,v in example.items()}
                this_example['term'] = len1 + len2
            else:
                nums = []
                for text in example['goal'] + example['sol1'] + example['sol2']:
                    tokens = tokenizer(text)['input_ids']
                    for token in tokens:
                        if token in vocabs:
                            nums.append(1/vocabs[token])
                        else:
                            nums.append(1)
                this_example = {k:v for k,v in example.items()}
                this_example['term'] = sum(nums)/len(nums)
            examples.append(this_example)
    examples.sort(key = lambda x : x['term'])
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--piqa_data_dir", default=None, type=str, required=True,
					    help="The model dir for eval")
    parser.add_argument("--out_dir", default=None, type=str, required=True,
					    help="The dir of data for eval")
    parser.add_argument("--task", default=None, type=str, required=True,
					    help="The dir of data for eval, should be one of: sim, len, overlap")
    parser.add_argument("--vocab_file", default=None, type=str,
					    help="The dir of the train file if we are computing vocab_overlap")                  
    args = parser.parse_args()
    if args.task not in ["sim", "len", "overlap"]:
        raise ValueError("Task doesn't exist")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space = True)
    examples = rank_quatiles(args.task, args.piqa_data_dir, tokenizer, args.vocab_file)
    l = len(examples)
    with open(args.out_dir + "/" + args.task[0] + "25.jsonl", "w")as f:
        for example in examples[:l//4]:
            f.write(json.dumps(example))
            f.write("\n")
    with open(args.out_dir + "/" + args.task[0] + "50.jsonl", "w")as f:
        for example in examples[l//4:2*l//4]:
            f.write(json.dumps(example))
            f.write("\n")
    with open(args.out_dir + "/" + args.task[0] + "75.jsonl", "w")as f:
        for example in examples[2*l//4:3*l//4]:
            f.write(json.dumps(example))
            f.write("\n")
    with open(args.out_dir + "/" + args.task[0] + "100.jsonl", "w")as f:
        for example in examples[3*l//4:]:
            f.write(json.dumps(example))
            f.write("\n")
    return 1

if __name__ == '__main__':
    main()