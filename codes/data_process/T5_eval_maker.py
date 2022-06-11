# this code is used to transfrom the raw 5 benchmarks into T5 forms data

import json
import argparse
import re
from tqdm import tqdm

def raw_to_T5(in_dir, out_dir):
    anli_data = []
    csqa_data = []
    piqa_data = []
    soqa_data = []
    wino_data = []
    with open(in_dir + "/piqa_dev.jsonl")as f:
        for row in tqdm(f):
            result = {}
            sample = json.loads(row)
            context = sample['goal']
            
            first_choice = "reasoning: " + context + " " + sample["sol1"]
            second_choice = "reasoning: " + context + " " + sample["sol2"]
            result['candidates'] = ["1", "1"]
            result['context'] = [first_choice, second_choice]
            result['correct'] = sample['label']
            piqa_data.append(result)

    with open(out_dir + "/piqa_dev_T5.jsonl", "w")as f:
        for data in tqdm(piqa_data):
            f.write(json.dumps(data)+"\n")


    with open(in_dir + "/anli_dev.jsonl")as f:
        for row in tqdm(f):
            result = {}
            sample = json.loads(row)
            context = sample['context']
            stem = sample["question"]['stem']
            first_choice = "reasoning: " + ' ' + context + ' ' + sample["question"]['choices'][0]['text'] + ' ' + stem
            second_choice = "reasoning: " + ' ' + context + ' ' + sample["question"]['choices'][1]['text'] + ' ' + stem
            result['context'] = [first_choice, second_choice]
            result['candidates'] = ["1", "1"]

            if sample['answerKey'] == 'A':
                result['correct'] = 0
            elif sample['answerKey'] == 'B':
                result['correct'] = 1
            else:
                continue
            anli_data.append(result)
    with open(out_dir + "/anli_dev_T5.jsonl", "w")as f:
        for data in tqdm(anli_data):
            f.write(json.dumps(data)+"\n")

    QUESTION_TO_ANSWER_PREFIX = {
        "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
        "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
        "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
        "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
        "What will (.*) do next?": r"[SUBJ] then",
        "How would (.*) feel after?": r"[SUBJ] then",
        "How would you describe (.*)?": r"[SUBJ] is seen as",
        "What kind of person is (.*)?": r"[SUBJ] is seen as",
        "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
        "Why did (.*) do that?": r"Before, [SUBJ] wanted",
        "Why did (.*) do this?": r"Before, [SUBJ] wanted",
        "Why did (.*) want to do this?": r"Before, [SUBJ] wanted",
        "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
        "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
        "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
        "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
        "What will happen to (.*)?": r"[SUBJ] then",
        "What will happen to (.*) next?": r"[SUBJ] then"
    }

    with open(in_dir + "/commonsenseqa_dev.jsonl")as f:
        for row in tqdm(f):
            result = {}
            sample = json.loads(row)
            context = sample['question']['stem']
            candidates = []
            candidates.append("reasoning: " + ' ' + context + ' ' + sample["question"]['choices'][0]['text'] + '.')
            candidates.append("reasoning: " + ' ' + context + ' ' + sample["question"]['choices'][1]['text'] + '.')
            candidates.append("reasoning: " + ' ' + context + ' ' +  sample["question"]['choices'][2]['text'] + '.')
            candidates.append("reasoning: " + ' ' + context + ' ' +  sample["question"]['choices'][3]['text'] + '.')
            candidates.append("reasoning: " + ' ' + context + ' ' +  sample["question"]['choices'][4]['text'] + '.')
            result['context'] = candidates
            result['candidates'] = ["1", "1", "1", "1", "1"]
            if sample['answerKey'] == 'A':
                result['correct'] = 0
            elif sample['answerKey'] == 'B':
                result['correct'] = 1
            elif sample['answerKey'] == 'C':
                result['correct'] = 2
            elif sample['answerKey'] == 'D':
                result['correct'] = 3
            elif sample['answerKey'] == 'E':
                result['correct'] = 4
            else:
                continue
            csqa_data.append(result)

    with open(out_dir + "/commonsenseqa_dev_T5.jsonl", "w")as f:
        for data in tqdm(csqa_data):
            f.write(json.dumps(data)+"\n")


    with open(in_dir + "/socialiqa_dev.jsonl")as f:
        for row in tqdm(f):
            result = {}
            sample = json.loads(row)
            context = sample['context']
            question = sample['question']
            answer_prefix = ""
            for template, ans_prefix in QUESTION_TO_ANSWER_PREFIX.items():
                m = re.match(template, question)
                if m is not None:
                    subj = m.group(1)
                    if subj.endswith('?'):
                        subj = subj[:-1]
                    answer_prefix = ans_prefix.replace("[SUBJ]", subj)
                    break

            if answer_prefix == "":
                answer_prefix = question.replace("?", "is")
            
            question = context + ' ' + answer_prefix
            choices = []
            for choice in [sample['answerA'], sample['answerB'], sample['answerC']]:
                if answer_prefix.endswith('wanted to') and choice.startswith('wanted to'):
                    choice = choice[9:].strip()
                if answer_prefix.endswith('needed to') and choice.startswith('needed to'):
                    choice = choice[9:].strip()
                if answer_prefix.endswith('to') and choice.startswith('to'):
                    choice = choice[2:].strip()
                choice = choice[0].lower() + choice[1:] + "."
                choices.append(choice)
            candidates = []
            candidates.append("reasoning: " + question + " " +  choices[0])
            candidates.append("reasoning: " + question + " " +  choices[1])
            candidates.append("reasoning: " + question + " " +  choices[2])
            result['context'] = candidates
            result['candidates'] = ["1", "1", "1"]
            if sample['correct'] == 'A':
                result['correct'] = 0
            elif sample['correct'] == 'B':
                result['correct'] = 1
            elif sample['correct'] == 'C':
                result['correct'] = 2
            else:
                continue
            soqa_data.append(result)

    with open(out_dir + "/socialiqa_dev_T5.jsonl", "w")as f:
        for data in tqdm(soqa_data):
            f.write(json.dumps(data)+"\n")


    with open(in_dir + "/winogrande_dev.jsonl")as f:
        for row in tqdm(f):
            result = {}
            sample = json.loads(row)
            context = sample['sentence']
            context = context.split("_")
            candidates = []
            candidates.append("reasoning: " + ' ' + context[0] + sample["option1"] + context[1])
            candidates.append("reasoning: " + ' ' + context[0] + sample["option2"] + context[1])
            result['context'] = candidates
            result['candidates'] = ["1", "1"]
            result['correct'] = int(sample['answer']) - 1
            wino_data.append(result)
    with open(out_dir + "/winogrande_dev_T5.jsonl", "w")as f:
        for data in tqdm(wino_data):
            f.write(json.dumps(data)+"\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_data_dir", default=None, type=str, required=True,
					    help="The model dir for eval")
    parser.add_argument("--out_dir", default=None, type=str, required=True,
					    help="The dir of data for eval")	
    args = parser.parse_args()
    in_dir = args.dev_data_dir
    out_dir = args.out_dir
    raw_to_T5(in_dir, out_dir)
    return 1

if __name__ == "__main__":
	main()