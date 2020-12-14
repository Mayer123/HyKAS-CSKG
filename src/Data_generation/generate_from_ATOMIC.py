import csv
import logging
from tqdm import tqdm
import json
import re
import ftfy
import random
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
skip_words = set(stopwords.words('english'))
skip_words.add('\'s')
skip_words.add('.')
skip_words.add(',')
import sys
sys.path.append('../')
sys.path.append('.')
import os
import argparse
from Training.data_utils import PERSON_NAMES
from sentence_transformers import SentenceTransformer, util
import pickle
import numpy as np
import torch

def text_standardize(text):
	"""
	Borrowed from COMET repo 
	"""
	text = text.replace('—', '-')
	text = text.replace('–', '-')
	text = text.replace('―', '-')
	text = text.replace('…', '...')
	text = text.replace('´', "'")
	text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
	text = re.sub(r'\s*\n\s*', ' \n ', text)
	text = re.sub(r'[^\S\n]+', ' ', text)
	return text.strip()

def overlap_exist(tail, keywords):
	tail = nltk.word_tokenize(tail.lower())
	if len(set(tail).intersection(keywords)) > 0:
		return True
	else:
		return False

def write_data(filename, data):
	with open(filename, 'w') as fout:
		for sample in data:
			fout.write(json.dumps(sample))
			fout.write('\n')

def read_data(filename):
	data = []
	with open(filename, 'r') as f:
		for line in f:
			data.append(json.loads(line))
	return data

class ATOMICProcessor(object):
	def __init__(self, args):
		self.mapping = {
		'xAttr' : '. PersonX is seen as',
		'xIntent' : '. Before, PersonX wanted',
		'xNeed' : '. Before, PersonX needed to',
		'xReact': '. As a result, PersonX felt',
		'xWant': '. As a result, PersonX wanted to',
		'xEffect': '. PersonX then',
		'oReact': '. As a result, others felt',
		'oWant': '. As a result, others wanted to',
		'oEffect': '. Others then'
		}
		self.xset = ['PersonX', 'Personx', 'personX', 'personx', 'Person X', 'Person x', 'person X', 'person x']
		self.yset = ['PersonY', 'Persony', 'personY', 'persony', 'Person Y', 'Person y', 'person Y', 'person y']
		self.zset = ['PersonZ', 'Personz', 'personZ', 'personz', 'Person Z', 'Person z', 'person Z', 'person z']
		self.xset1 = [' X ', ' x ', ' X\'', ' x\'', ' X.', ' x.']
		self.yset1 = [' Y ', ' y ', ' Y\'', ' y\'', ' Y.', ' y.']
		self.zset1 = [' Z ', ' z ', ' Z\'', ' z\'', ' Z.', ' z.']
		self.answerKey_mapping = {}
		self.D = [[], []]
		self.labels=[]
		self.filelist = [args.train_KG, args.dev_KG]
		self.tail_keywords = defaultdict(set)
		self.adv = False

	def get_person_set(self, context):
		person_set = []
		if any([x in context for x in self.xset+self.xset1]):
			person_set += self.xset+self.xset1
		if any([y in context for y in self.yset+self.yset1]):
			person_set += self.yset+self.yset1
		if any([z in context for z in self.zset+self.zset1]):
			person_set += self.zset+self.zset1
		return person_set

	def find_underscore_length(self, seq):
		start = "_"
		while start in seq:
			start += "_"
		return start[:-1]

	def fill_names(self, sent, names):
		for x in self.xset:
			sent = sent.replace(x, names[0])
		for x in self.xset1:
			sent = sent.replace(x, x[0]+names[0]+x[-1])
		for y in self.yset:
			sent = sent.replace(y, names[1])
		for y in self.yset1:
			sent = sent.replace(y, y[0]+names[0]+y[-1])
		for z in self.zset:
			sent = sent.replace(z, names[2])
		for z in self.zset1:
			sent = sent.replace(z, z[0]+names[0]+z[-1])
		return sent

	def fix_templates(self, context, tail):
		if context.endswith('wanted to') and tail.startswith('wanted to'):
			tail = tail[9:].strip()
		if context.endswith('needed to') and tail.startswith('needed to'):
			tail = tail[9:].strip()
		if context.endswith('to') and tail.startswith('to'):
			tail = tail[2:].strip()
		if len(tail) != 0:
			tail = tail[0].lower()+tail[1:]
			if not tail.endswith('.'):
				tail += '.'
		return tail

	def negative_sample(self, prefix, dim, correct_ones, data, person_set, question, correct_answer):
		negatives = []
		while len(negatives) < 2:
			sample = random.choice(data)
			if len(sample[1][dim]) == 0:
				continue
			neg = random.choice(sample[1][dim])
			if len(set(prefix).intersection(self.tail_keywords[(neg, dim)])) != 0:
				continue
			if neg in correct_ones:
				continue
			if neg in negatives:
				continue
			if neg[:-1] in correct_answer[:-1].split() or correct_answer[:-1] in neg[:-1].split():
				continue
			if len(person_set) < len(self.xset+self.xset1)*2 and any([y in neg for y in self.yset+self.yset1]):
				continue
			if len(person_set) < len(self.xset+self.xset1)*3 and any([z in neg for z in self.zset+self.zset1]):
				continue
			negatives.append(neg)
		return negatives

	def create_dataset(self, data):
		generated_data = []
		count = 0
		for sample in tqdm(data):
			for k, v in sample[1].items():
				if len(v) != 0:
					context = text_standardize(ftfy.fix_text(sample[0]))
					person_set = self.get_person_set(context)
					question = self.mapping[k]
					for vv in v:
						correct_answer = vv
						if overlap_exist(correct_answer, sample[-1]):
							continue
						negative_answers = self.negative_sample(sample[-1], k, v, data, person_set, context+question, correct_answer)
						if negative_answers == None:
							continue
						names = random.sample(PERSON_NAMES, 3)
						new_context = self.fill_names(context+question, names)
						correct_answer = self.fill_names(correct_answer, names)
						negative_answers = [self.fill_names(neg, names) for neg in negative_answers]
						candidates = negative_answers+[correct_answer]
						random.shuffle(candidates)
						label = candidates.index(correct_answer)
						count += 1
						generated_data.append({'id':str(count), 'dim':k, 'context':new_context, 'correct':label, 'candidates':candidates, 'keywords': sample[-1]})
		return generated_data

	def get_train_examples(self):
		self.load_data(self.filelist[0], 0)
		return self.create_dataset(self.D[0])

	def get_dev_examples(self):
		self.load_data(self.filelist[1], 1)
		return self.create_dataset(self.D[1])

	def load_data(self, filename, sid):
		skipped = 0
		previous = 'random stuff'
		prefix = 'random stuff'
		cache = None
		with open(filename, "r") as f:
			csvreader = csv.reader(f)
			fields = next(csvreader)
			for row in tqdm(csvreader):
				if row[0] != previous:
					if cache != None:
						self.D[sid].append([previous, cache, prefix])
					previous = row[0]
					cache = {k:[] for k, v in self.mapping.items()}
				row[1:-1] = [json.loads(e) for e in row[1:-1]]
				prefix = row[-2]
				for i, attr in enumerate(row[1:-2]):
					for ending in attr:
						ending = ending.lower()
						ending = self.fix_templates(self.mapping[fields[i+1]], text_standardize(ftfy.fix_text(ending)))
						if '_' in ending:
							tok = self.find_underscore_length(ending)
							ending = ending.replace(tok, "___")
						if ending != 'none.' and len(ending) > 0 and ending not in cache[fields[i+1]]:
							self.tail_keywords[(ending, fields[i+1])] |= set(prefix)
							cache[fields[i+1]].append(ending)
			if cache != None:
				self.D[sid].append([previous, cache, prefix])
			print (len(self.D[sid]))

class ATOMICAdvAnswerProcessor(ATOMICProcessor):
	def __init__(self, args):
		super(ATOMICAdvAnswerProcessor, self).__init__(args)
		with open(os.path.join(args.out_dir, 'atomic_tails.pkl'), "rb") as fin:
			d = pickle.load(fin)
		self.tail_index = d['sentences']
		self.reverse_tail_index = {v:k for k, v in self.tail_index.items()}
		self.embeddings = d['embeddings']
		self.lower_bounds = Counter()
		self.high_prob = 0.4
		self.low_prob = 0.3
		self.patience = 10
		self.step_size = 0.05
		self.downsample_size = 50
		self.adv = True

	def negative_sample(self, prefix, dim, correct_ones, data, person_set, question, correct_answer):
		negatives = []
		curr_data = random.choices(data, k=self.downsample_size)	
		distractors = list(set([neg for sample in curr_data for neg in sample[1][dim]]))
		distractors = [neg for neg in distractors if len(set(prefix).intersection(self.tail_keywords[(neg, dim)])) == 0]
		distractors_mapping = {i:self.tail_index[neg] for i, neg in enumerate(distractors)}
		distractors_indices = list(distractors_mapping.values())
		distractor_emb = self.embeddings[distractors_indices]
		correct_emb = self.embeddings[self.tail_index[correct_answer]]
		cos_scores = util.pytorch_cos_sim(correct_emb, distractor_emb)[0]
		high_prob = self.high_prob
		low_prob = self.low_prob
		midpoint = np.argwhere((cos_scores.numpy()>low_prob) & (cos_scores.numpy() < high_prob)).squeeze(1)
		midinf = 0
		while len(midpoint) < self.patience and midinf < self.patience:
			midinf += 1
			low_prob -= self.step_size
			midpoint = np.argwhere((cos_scores.numpy()>low_prob) & (cos_scores.numpy() < high_prob)).squeeze(1)
		if len(midpoint) == 0:
			print ('empty')
			return None
		infinite = 0
		while len(negatives) < 2 and infinite < self.patience:
			infinite += 1
			sample_idx = random.choice(midpoint)
			neg = self.reverse_tail_index[distractors_mapping[sample_idx.item()]]
			if neg in correct_ones:
				continue
			if neg in negatives:
				continue
			if neg[:-1] in correct_answer[:-1].split() or correct_answer[:-1] in neg[:-1].split():
				continue
			if len(person_set) < len(self.xset+self.xset1)*2 and any([y in neg for y in self.yset+self.yset1]):
				continue
			if len(person_set) < len(self.xset+self.xset1)*3 and any([z in neg for z in self.zset+self.zset1]):
				continue
			negatives.append(neg)
		self.lower_bounds[low_prob] += 1
		if len(negatives) < 2:
			return None
		return negatives

class ATOMICAdvQuestionProcessor(ATOMICProcessor):
	def __init__(self, args):
		super(ATOMICAdvQuestionProcessor, self).__init__(args)
		with open(os.path.join(args.out_dir, 'atomic_tails.pkl'), "rb") as fin:
			d = pickle.load(fin)
		self.tail_index = d['sentences']
		self.reverse_tail_index = {v:k for k, v in self.tail_index.items()}
		self.tail_embeddings = d['embeddings']
		with open(os.path.join(args.out_dir, 'atomic_heads.pkl'), "rb") as fin:
			d = pickle.load(fin)
		self.head_index = d['sentences']
		self.revers_head_index = {v:k for k, v in self.head_index.items()}
		self.head_embeddings = d['embeddings']
		self.lower_bounds = Counter()
		self.high_prob = 0.4
		self.low_prob = 0.3
		self.patience = 10
		self.step_size = 0.05
		self.downsample_size = 200
		self.adv = True

	def negative_sample(self, prefix, dim, correct_ones, data, person_set, question, correct_answer):
		negatives = []
		curr_data = random.choices(data, k=self.downsample_size)	
		distractors = list(set([neg for sample in curr_data for neg in sample[1][dim]]))
		distractors = [neg for neg in distractors if len(set(prefix).intersection(self.tail_keywords[(neg, dim)])) == 0]
		distractors_mapping = {i:self.tail_index[neg] for i, neg in enumerate(distractors)}
		distractors_indices = list(distractors_mapping.values())
		distractor_emb = self.tail_embeddings[distractors_indices]
		question_emb = self.head_embeddings[self.head_index[question]]
		cos_scores = util.pytorch_cos_sim(question_emb, distractor_emb)[0]
		high_prob = self.high_prob
		low_prob = self.low_prob
		midpoint = np.argwhere((cos_scores.numpy()>low_prob) & (cos_scores.numpy() < high_prob)).squeeze(1)
		midinf = 0
		while len(midpoint) < self.patience and midinf < self.patience:
			midinf += 1
			low_prob -= self.step_size
			midpoint = np.argwhere((cos_scores.numpy()>low_prob) & (cos_scores.numpy() < high_prob)).squeeze(1)
		if len(midpoint) == 0:
			print ('empty')
			return None
		infinite = 0
		while len(negatives) < 2 and infinite < self.patience:
			infinite += 1
			sample_idx = random.choice(midpoint)
			neg = self.reverse_tail_index[distractors_mapping[sample_idx.item()]]
			if neg in correct_ones:
				continue
			if neg in negatives:
				continue
			if neg[:-1] in correct_answer[:-1].split() or correct_answer[:-1] in neg[:-1].split():
				continue
			if len(person_set) < len(self.xset+self.xset1)*2 and any([y in neg for y in self.yset+self.yset1]):
				continue
			if len(person_set) < len(self.xset+self.xset1)*3 and any([z in neg for z in self.zset+self.zset1]):
				continue
			negatives.append(neg)
		self.lower_bounds[low_prob] += 1
		if len(negatives) < 2:
			return None
		return negatives

def build_embeddings_answers(args):
	if os.path.exists(os.path.join(args.out_dir, 'atomic_tails.pkl')):
		print ('tail embeddings already exist, skip computation')
		return 
	processor = ATOMICProcessor(args)
	model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
	all_tails = {}
	files = [args.train_KG, args.dev_KG]
	for file in files:
		with open(file, 'r') as f:
			csvreader = csv.reader(f)
			fields = next(csvreader)
			for row in tqdm(csvreader):
				row[1:-1] = [json.loads(e) for e in row[1:-1]]
				for i, attr in enumerate(row[1:-2]):
					for ending in attr:
						ending = ending.lower()
						if ending != 'none':
							tail = text_standardize(ftfy.fix_text(ending))
							tail = processor.fix_templates(processor.mapping[fields[i+1]], tail)
							if '_' in tail:
								tok = processor.find_underscore_length(tail)
								tail = tail.replace(tok, "___")
							if tail not in all_tails:
								all_tails[tail] = len(all_tails)
	print (len(all_tails))
	corpus = [k for k, v in all_tails.items()]
	embeddings = model.encode(corpus, show_progress_bar=True, device=0, num_workers=4)
	print (len(embeddings), embeddings.shape)
	with open(os.path.join(args.out_dir, 'atomic_tails.pkl'), "wb") as fOut:
		pickle.dump({'sentences': all_tails, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

def build_embeddings_question(args):
	if os.path.exists(os.path.join(args.out_dir, 'atomic_heads.pkl')):
		print ('head embeddings already exist, skip computation')
		return 
	processor = ATOMICProcessor(args)
	model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
	all_heads = {}
	files = [args.train_KG, args.dev_KG]
	previous = 'random stuff'
	for file in files:
		with open(file, 'r') as f:
			csvreader = csv.reader(f)
			fields = next(csvreader)
			for row in tqdm(csvreader):
				row[1:-1] = [json.loads(e) for e in row[1:-1]]
				if row[0] != previous:
					previous = row[0]
					head = text_standardize(ftfy.fix_text(row[0]))
					for i, attr in enumerate(row[1:-2]):
						rel = processor.mapping[fields[i+1]]
						question = head + rel
						if question not in all_heads:
							all_heads[question] = len(all_heads)
				
	print (len(all_heads))
	corpus = list(all_heads.keys())
	embeddings1 = model.encode(corpus[:100000], show_progress_bar=True, device=0, num_workers=4)
	embeddings2 = model.encode(corpus[100000:], show_progress_bar=True, device=0, num_workers=4)
	embeddings = np.concatenate([embeddings1, embeddings2], axis=0)
	print (len(embeddings), embeddings.shape)
	with open(os.path.join(args.out_dir, 'atomic_heads.pkl'), "wb") as fOut:
		pickle.dump({'sentences': all_heads, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_KG", default=None, type=str, required=True, help="ATOMIC train file")
	parser.add_argument("--dev_KG", default=None, type=str, required=True, help="ATOMIC dev file")
	parser.add_argument("--strategy", default='random', type=str, required=False, choices=['random', 'adv-answer', 'adv-question'], help="which data generation strategy to use")
	parser.add_argument("--out_dir", default=None, type=str, required=True, help="Output dir")
	parser.add_argument('--do_split', action="store_true", help="Further split training set into subsets for AFLite")
	args = parser.parse_args()
	random.seed(1)
	np.random.seed(1)
	if args.strategy == 'random':
		processor = ATOMICProcessor(args)
	elif args.strategy == 'adv-answer':
		print ('Using adv-answer strategy')
		build_embeddings_answers(args)
		processor = ATOMICAdvAnswerProcessor(args)
	elif args.strategy == 'adv-question':
		print ('Using adv-question strategy')
		build_embeddings_answers(args)
		build_embeddings_question(args)
		processor = ATOMICAdvQuestionProcessor(args)
	else:
		print ('strategy not recognized')
		exit(0)
	dev_examples = processor.get_dev_examples()
	write_data(os.path.join(args.out_dir, 'dev_'+args.strategy+'.jsonl'), dev_examples)
	train_examples = processor.get_train_examples()
	write_data(os.path.join(args.out_dir, 'train_'+args.strategy+'.jsonl'), train_examples)
	if args.do_split:
		assert args.strategy == 'random'
		random.shuffle(train_examples)
		print ('splitting train into subsets, which can be used for AFLite (only valid for random strategy)')
		train_examples_1 = train_examples[:int(len(train_examples)*0.01)]
		train_examples_4 = train_examples[int(len(train_examples)*0.01):int(len(train_examples)*0.05)]
		train_examples_95 = train_examples[int(len(train_examples)*0.05):]
		write_data(os.path.join(args.out_dir, 'train_1%_'+args.strategy+'.jsonl'), train_examples_1)
		write_data(os.path.join(args.out_dir, 'train_4%_'+args.strategy+'.jsonl'), train_examples_4)
		write_data(os.path.join(args.out_dir, 'train_95%_'+args.strategy+'.jsonl'), train_examples_95)




