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
from transformers import AutoTokenizer, AutoModelWithLMHead

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
        choices = [fields["sol1"], fields["sol2"]]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice[0].lower() + choice[1:]}" for choice in choices]
        return context, question, label, choices, context_with_choices


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

        choices = [
            " ".join((answer_prefix, choice[0].lower() + choice[1:])).replace(
                "?", "").replace("wanted to wanted to", "wanted to").replace(
                "needed to needed to", "needed to").replace("to to", "to") for choice in choices]

        context_with_choices = [f"{context} {choice}" for choice in choices]
        return context, question, label, choices, context_with_choices

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
        context_with_choices = [f"{question} {choice}" for choice in choices]
        return context, question, label, choices, context_with_choices

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
        context_with_choices = [f"{question} {choice}" for choice in choices]
        return context, question, label, choices, context_with_choices

class WinograndeInstanceReader(InstanceReader):
    """
    Reads the WinoGrande dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = fields['sentence']
        if not context.endswith("."):
            context += "."

        label = fields['answer']
        choices = [fields['option1'], fields['option2']]
        label = int(label) - 1
        question = ''
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [context.replace("_", choice) for choice in choices]
        return context, question, label, choices, context_with_choices


class CommonsenseqaInstanceReader(InstanceReader):
    """
    Reads the CommonsenseQA dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = ''

        question = 'Q: ' + fields['question']['stem']
        label = ['A','B','C','D','E'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = ['A: '+ c['text'] for c in fields['question']['choices']]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice[0].lower() + choice[1:]}" for choice in choices]
        return context, question, label, choices, context_with_choices

class ANLIInstanceReader(InstanceReader):
    """
    Reads the aNLI dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        label = ['A','B'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = [c['statement'] for c in fields['statements']]
        return label, choices

    @overrides
    def fields_to_instance(self, fields):
        label, choices = self.to_uniform_fields(fields)
        return None, None, label, None, choices

INSTANCE_READERS = {"socialiqa": SocialIQAInstanceReader,
                    "winogrande": WinograndeInstanceReader,
                    "piqa": PiqaInstanceReader,
                    "commonsenseqa":CommonsenseqaInstanceReader,
                    "anli": ANLIInstanceReader,
                    "atomic": ATOMICInstanceReader,
                    'cwwv': CWWVInstanceReader}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="gpt2-large", type=str, required=False, help="language model to use")
    parser.add_argument("--dataset_file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")
    parser.add_argument("--cache_dir", default=None, type=str, required=False, help="where the model is cached")
    parser.add_argument("--reader", default=None, type=str, required=True, help="which reader to use")
    args = parser.parse_args()
    logger.info(args)

    task = args.reader 
    if args.lm != 'gpt2-large':
        model_path = ['gpt2']+args.lm.split('/')[-1:]+[task]
        model_path = '_'.join([m for m in model_path if m != ''])
        out_dir = os.path.join(args.out_dir, model_path)
    else:
        out_dir = os.path.join(args.out_dir, 'gpt2_'+task)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Load the language model
    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    model, tokenizer = init_model(args.lm, device, args.cache_dir)

    # Load the dataset
    instance_reader = INSTANCE_READERS[args.reader]()
    
    out_file = os.path.join(out_dir, "predictions.jsonl")
    log_file = os.path.join(out_dir, 'results.txt')
    gold = []
    predictions = []
    results = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    sample_id = 0
    # Predict instances
    with open(out_file, "w") as f_out:
        with open(args.dataset_file) as f_in:
            for line in tqdm.tqdm(f_in):
                fields = json.loads(line.strip())
                context, question, label, choices, context_with_choices = \
                    instance_reader.fields_to_instance(fields)
                if sample_id == 0:
                    results.append(json.dumps(context_with_choices))
                gold.append(label)
                # Tokenize and pad
                tokenized = [tokenizer.encode(text) for text in context_with_choices]
                max_length = max([len(text) for text in tokenized])
                att_mask = torch.zeros((len(tokenized), max_length)).to(device)
                for i in range(len(tokenized)):
                    att_mask[i][:len(tokenized[i])] = 1
                tokenized = [text + [pad_token_id] * (max_length - len(text)) for text in tokenized]
                tokenized = torch.tensor(tokenized).long().to(device)
                prediction = int(np.argmin(get_lm_score(model, tokenized, pad_token_id, att_mask)))
                fields["prediction"] = prediction
                predictions.append(prediction)
                f_out.write(json.dumps(fields) + "\n")
                sample_id += 1

    # Don't report accuracy if we don't have the labels
    if None not in gold:
        accuracy = (np.array(gold)==np.array(predictions)).mean()
        print(f"Accuracy: {accuracy:.3f}")
        results.append(f"Accuracy : {accuracy:.3f}")
    with open(log_file, 'w') as fout:
        for line in results:
            fout.write(line + '\n')


def get_lm_score(model, batch, pad_token_id, att_mask):
    """
    Get the cross entropy loss of the texts in batch using the langage model
    """
    # Batch: [num_choices, max_length]
    with torch.no_grad():
        num_choices, max_length = batch.shape
        shift_labels = batch[..., 1:].contiguous().view(-1)
        lm_logits = model(batch, attention_mask=att_mask)[0]
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(num_choices, -1).sum(1).cpu().numpy()
        valid_tokens = (batch!=pad_token_id).long().sum(1).cpu().numpy()
        loss /= valid_tokens 
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelWithLMHead.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == '__main__':
    main()
