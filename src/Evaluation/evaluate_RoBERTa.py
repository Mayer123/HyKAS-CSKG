import os
import re
import json
import tqdm
import torch
import logging
import argparse
import numpy as np
from overrides import overrides
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaForMaskedLM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_SEQUENCE_PER_TIME = 80

class InstanceReader(object):
    def to_uniform_fields(self, fields):
        pass

    def fields_to_instance(self, fields):
        pass

class PiqaInstanceReader(InstanceReader):
    """
    Reads the PIQA dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = ""
        question = fields["goal"]
        label = fields.get('label', None)
        choices = [fields["sol1"][0].lower()+fields["sol1"][1:], fields["sol2"][0].lower()+fields["sol2"][1:]]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        return context, question, label, choices


class SocialIQAInstanceReader(InstanceReader):
    """
    Reads the SocialIQa dataset into a unified format with context, question, label, and choices.
    """
    def __init__(self):
        super(SocialIQAInstanceReader).__init__()
        self.QUESTION_TO_ANSWER_PREFIX = {
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

    @overrides
    def to_uniform_fields(self, fields):
        context = fields['context']
        if not context.endswith("."):
            context += "."

        question = fields['question']
        label = fields['correct']
        choices = [fields['answerA'], fields['answerB'], fields['answerC']]
        choices = [c + "." if not c.endswith(".") else c for c in choices]
        label = ord(label) - 65
        return context, question, label, choices

    def convert_choice(self, choice, answer_prefix):
        if answer_prefix.endswith('wanted to') and choice.startswith('wanted to'):
            choice = choice[9:].strip()
        if answer_prefix.endswith('needed to') and choice.startswith('needed to'):
            choice = choice[9:].strip()
        if answer_prefix.endswith('to') and choice.startswith('to'):
            choice = choice[2:].strip()
        choice = choice[0].lower() + choice[1:]
        return choice

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)

        answer_prefix = ""
        for template, ans_prefix in self.QUESTION_TO_ANSWER_PREFIX.items():
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
        choices = [self.convert_choice(choice, answer_prefix) for choice in choices]

        return context, question, label, choices

class ATOMICInstanceReader(InstanceReader):
    """
    Reads the ATOMIC dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        question = fields['context']
        label = fields['correct']
        choices = [fields['candidates'][0], fields['candidates'][1], fields['candidates'][2]]
        return '', question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        #print (question, choices)
        return context, question, label, choices

class CWWVInstanceReader(InstanceReader):
    """
    Reads the CWWV dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        question = fields['question']['stem']
        if question.endswith('.'):
            question = question[:-1]
        if not question.endswith('[MASK]'):
            print ('should not happen')
            exit(0)
        question = question[:-7]
        label = ['A','B','C'].index(fields['answerKey'])
        choices = [fields['question']['choices'][0]['text']+'.', fields['question']['choices'][1]['text']+'.', fields['question']['choices'][2]['text']+'.']
        return '', question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        return context, question, label, choices

class WinograndeInstanceReader(InstanceReader):
    """
    Reads the WinoGrande dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = fields['sentence']
        if not context.endswith("."):
            context += "."
        context = context.split('_')
        label = fields['answer']
        choices = [fields['option1']+context[1], fields['option2']+context[1]]
        label = int(label) - 1
        question = context[0].strip()
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        return context, question, label, choices


class CommonsenseqaInstanceReader(InstanceReader):
    """
    Reads the CommonsenseQA dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = ''
        question = 'Q: '+ fields['question']['stem']
        label = ['A','B','C','D','E'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = ['A: '+c['text'][0].lower()+c['text'][1:] for c in fields['question']['choices']]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        return context, question, label, choices

class ANLIInstanceReader(InstanceReader):
    """
    Reads the aNLI dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = ''
        question = fields['context']
        label = ['A','B'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = [c['text']+' '+fields['question']['stem'] for c in fields['question']['choices']]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        return context, question, label, choices

INSTANCE_READERS = {"socialiqa": SocialIQAInstanceReader,
                    "winogrande": WinograndeInstanceReader,
                    "piqa": PiqaInstanceReader,
                    "commonsenseqa":CommonsenseqaInstanceReader,
                    "anli": ANLIInstanceReader,
                    'atomic': ATOMICInstanceReader,
                    'cwwv': CWWVInstanceReader}

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

def score_task(question, choices, tokenizer, device, model):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    question_ids = tokenizer.encode(question)
    choice_ids = [tokenizer.encode(choice, add_prefix_space=True)[1:-1] for choice in choices]
    sequences = [question_ids[:-1] + choice_ids[i] +[tokenizer.sep_token_id] for i in range(len(choice_ids))]
    label_ids = [[-100]+text[1:-1]+[-100] for text in sequences]
    sequences, label_ids, attention_mask = prepare_input(sequences, label_ids, pad_token_id)
    prediction = token_wise_scoring(sequences, label_ids, attention_mask, tokenizer, device, model)
    return prediction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="roberta-large", type=str, required=False, help="language model to use")
    parser.add_argument("--dataset_file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")
    parser.add_argument("--cache_dir", default=None, type=str, required=False, help="where the model is cached")
    parser.add_argument("--reader", default=None, type=str, required=True, help="which reader to use")
    args = parser.parse_args()
    logger.info(args)
    task = args.reader
    if args.lm != 'roberta-large':
        model_path = ['roberta']+args.lm.split('/')[-1:]+[task]
        model_path = '_'.join([m for m in model_path if m != ''])
        out_dir = os.path.join(args.out_dir, model_path)
    else:
        out_dir = os.path.join(args.out_dir, 'roberta_'+task)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, 'predictions.jsonl')
    log_file = os.path.join(out_dir, 'results.txt')

    # Load the language model
    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    model, tokenizer = init_model(args.lm, device, args.cache_dir)

    # Load the dataset
    instance_reader = INSTANCE_READERS[args.reader]()
    
    gold = []
    predictions = []
    results = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    print ('currently evaluating the task', task)
    # Predict instances
    sample_id = 0
    with open(out_file, "w") as f_out:
        with open(args.dataset_file) as f_in:
            for line in tqdm.tqdm(f_in):
                fields = json.loads(line.strip())
                context, question, label, choices = \
                    instance_reader.fields_to_instance(fields)
                gold.append(label)
                if sample_id == 0:
                    results.append(json.dumps(context))
                    results.append(json.dumps(question))
                    results.append(json.dumps(choices))
                prediction = score_task(question, choices, tokenizer, device, model)
                fields["prediction"] = prediction
                predictions.append(prediction)
                f_out.write(json.dumps(fields) + "\n")
                sample_id += 1
    # Don't report accuracy if we don't have the labels
    if None not in gold:
        accuracy = (np.array(gold)==np.array(predictions)).mean()
        print(f"Accuracy: {accuracy:.3f}")
        results.append(f"Accuracy: {accuracy:.3f}")
    with open(log_file, 'w') as fout:
        for line in results:
            fout.write(line + '\n')

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
    logger.info(f'Initializing {model_name}')
    tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = RobertaForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    return model, tokenizer

if __name__ == '__main__':
    main()
