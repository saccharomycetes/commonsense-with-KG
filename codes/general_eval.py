import os
import re
import json
from tqdm import tqdm
import torch
import argparse
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaForMaskedLM

MAX_SEQUENCE_PER_TIME = 80

def token_wise_scoring(sequences, label_ids, attention_mask, tokenizer, device, model):
    choice_loss = [0 for i in range(len(sequences))]
    for i in range(len(sequences)):
        tmp_seq_list = []
        tmp_label_list = []
        tmp_attention_mask = []
        curr_label_ids = label_ids[i]
        for j, t in enumerate(curr_label_ids):
            if t == -100:
                continue
            tmp_seq = torch.tensor(sequences[i][:j]+[tokenizer.mask_token_id]+sequences[i][j+1:]).long().to(device)
            tmp_label = torch.tensor([-100]*j+sequences[i][j:j+1]+[-100]*(len(sequences[i])-j-1)).long().to(device)
            tmp_seq_list.append(tmp_seq)
            tmp_label_list.append(tmp_label)
            tmp_attention_mask.append(torch.tensor(attention_mask[i]).long().to(device))
        tmp_seq_list = torch.stack(tmp_seq_list)
        tmp_label_list = torch.stack(tmp_label_list)
        tmp_attention_mask = torch.stack(tmp_attention_mask)
        if len(tmp_seq_list) < MAX_SEQUENCE_PER_TIME:
            loss = get_lm_score(model, tmp_seq_list, tmp_label_list, tmp_attention_mask)
        else:
            loss = []
            for chunk in range(0, len(tmp_seq_list), MAX_SEQUENCE_PER_TIME):
                loss.append(get_lm_score(model, tmp_seq_list[chunk:chunk+MAX_SEQUENCE_PER_TIME], tmp_label_list[chunk:chunk+MAX_SEQUENCE_PER_TIME], tmp_attention_mask[chunk:chunk+MAX_SEQUENCE_PER_TIME]))
            loss = np.concatenate(loss)
        choice_loss[i] = sum(loss)/len(loss) 
    prediction = choice_loss.index(min(choice_loss))
    return prediction

def prepare_input(sequences, label_ids, pad_token_id):
    max_length = max([len(text) for text in sequences])
    attention_mask = np.zeros((len(sequences), max_length))
    for i in range(len(sequences)):
        attention_mask[i][:len(sequences[i])] = 1
    sequences = [text + [pad_token_id] * (max_length - len(text)) for text in sequences]
    label_ids = [text + [-100] * (max_length - len(text)) for text in label_ids]
    return sequences, label_ids, attention_mask

def score_task(choices, tokenizer, device, model):

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # question_ids = tokenizer.encode(question)
    choice_ids = [tokenizer.encode(choice, add_prefix_space=True)[1:-1] for choice in choices]
    sequences = [choice_ids[i] +[tokenizer.sep_token_id] for i in range(len(choice_ids))]
    label_ids = [[-100]+text[1:-1]+[-100] for text in sequences]
    sequences, label_ids, attention_mask = prepare_input(sequences, label_ids, pad_token_id)
    prediction = token_wise_scoring(sequences, label_ids, attention_mask, tokenizer, device, model)

    return prediction

def get_lm_score(model, batch, label_ids, attention_mask):
    """
    Get the cross entropy loss of the texts in batch using the langage model
    """
    # Batch: [num_choices, max_length]
    with torch.no_grad():
        num_choices, max_length = batch.shape
        label_ids = label_ids.view(-1)
        lm_logits = model(batch, attention_mask=attention_mask)[0]
        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(lm_logits, label_ids)
        loss = loss.view(num_choices, -1).sum(1).cpu().numpy()
    return loss


def init_model(model_name: str,
               device: torch.device, cache_dir):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = RobertaForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="roberta-large", type=str, required=False, help="language model to use")
    parser.add_argument("--dataset_file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default='.', type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")
    args = parser.parse_args()

    model = RobertaForMaskedLM.from_pretrained(args.lm)
    tokenizer = RobertaTokenizer.from_pretrained(args.lm)
    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")

    with open(args.dataset_file) as f_in:
        testing_data = json.load(f_in)

    predictions = []
    for choices in tqdm(testing_data):
        prediction = score_task(choices, tokenizer, device, model)
        predictions.append(prediction)

    with open(os.path.join(args.out_dir, 'predictions.json'), 'w') as fout:
        json.dump(predictions, fout, indent=4)


if __name__ == '__main__':
    main()