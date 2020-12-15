from collections import Counter, defaultdict
import argparse
import sys
import random
import json
from tqdm import tqdm
import pickle as pkl
from string import Template
import numpy as np
from sentence_transformers import SentenceTransformer, util
from os import path
random.seed(1)
num_distractors=2

good_relations=['/r/Causes', '/r/UsedFor', '/r/CapableOf', '/r/CausesDesire', '/r/IsA', '/r/SymbolOf', '/r/MadeOf', '/r/LocatedNear', '/r/Desires', '/r/AtLocation', '/r/HasProperty', '/r/PartOf', '/r/HasFirstSubevent', '/r/HasLastSubevent'] 

q_sources=set(['CN', 'WD', 'WN'])
dist_only_sources=set(['VG'])

def format_question(q, a, distractors, q_id, head_label, template, source, rel):
	q_entry={}
	q_entry['id']=q_id
	q_entry['question']={'stem': q}

	answer_key=random.choice(["A", "B", "C"])
	q_entry["answerKey"]=answer_key
	if answer_key=="A":
		correct_option={"text": a, "label": "A"}
		dist1={"text": distractors[0], "label": "B"}
		dist2={"text": distractors[1], "label": "C"}
		options=[correct_option, dist1, dist2]
	elif answer_key=="B":
		correct_option={"text": a, "label": "B"}
		dist1={"text": distractors[0], "label": "A"}
		dist2={"text": distractors[1], "label": "C"}
		options=[dist1, correct_option, dist2]
	elif answer_key=="C":
		correct_option={"text": a, "label": "C"}
		dist1={"text": distractors[0], "label": "A"}
		dist2={"text": distractors[1], "label": "B"}
		options=[dist1, dist2, correct_option]
	q_entry["question"]["choices"]=options
	q_entry["question"]["head"]=head_label
	q_entry["question"]["source"]=source
	q_entry["question"]["template"]=template
	q_entry["question"]["relation"]=rel
	return q_entry

def select_distractors_noaf(data, head_label, heads, correct_answer, rel):
	"""Distractors without AFiltering"""

	negatives = []

	answer_heads=set(head_label.split())

	candidates=random.choices(list(data), k=num_distractors*100)

	for neg in candidates:
		distractor_heads=heads[(neg, rel)]
		if neg not in negatives and neg!=correct_answer and neg not in correct_answer and correct_answer not in neg and not (distractor_heads & answer_heads):
			negatives.append(neg)
			if len(negatives)>=num_distractors:
				return negatives, -1
	print('Not enough')
	return None, -1

def select_distractors_af(data, head_label, heads, correct_answer, rel, question, embeddings, sentence2id, sentences, q_or_a='a'):
	"""Distractors with AF"""
	high_prob = 0.6
	low_prob = 0.5
	step=0.05
	limit_dists=10
	if q_or_a=='q':
		downsample_size=num_distractors*400
	else:
		downsample_size=num_distractors*100

	negatives = []

	answer_heads=set(head_label.split())

	candidates=random.choices(list(data), k=downsample_size)

	distractors_indices = [sentence2id[sent] for sent in candidates] # todo!
	if q_or_a=='a':
		compare_index=sentence2id[correct_answer] # todo!
	else: # q
		compare_index=sentence2id[question_to_sentence(question)]
	dist_mapping = {i:val for i, val in enumerate(distractors_indices)}
	distractor_emb = embeddings[distractors_indices]
	correct_emb = embeddings[compare_index]
	cos_scores = util.pytorch_cos_sim(correct_emb, distractor_emb)[0]
	midpoint = np.argwhere((cos_scores.numpy()>low_prob) & (cos_scores.numpy() < high_prob)).squeeze(1)
	while len(midpoint) < limit_dists:
		low_prob -= step
		midpoint = np.argwhere((cos_scores.numpy()>low_prob) & (cos_scores.numpy() < high_prob)).squeeze(1)

	x=0
	while len(negatives) < num_distractors:
		if x>=10: 
			negatives=None
			print('Not enough')
			break	
		sample_idx = random.choice(midpoint)
		neg = sentences[dist_mapping[sample_idx.item()]]
		distractor_heads=heads[(neg, rel)]
		if neg not in negatives and neg!=correct_answer and neg not in correct_answer and correct_answer not in neg and not (distractor_heads & answer_heads):
			negatives.append(neg)
		x+=1
	return negatives, low_prob

def construct_from_template(h, r):
	t={
		"/r/Causes": "$node1 can cause [MASK]",
		"/r/UsedFor": "$node1 can be used for [MASK]",
		"/r/CapableOf": "$node1 is capable of [MASK]", 
		"/r/CausesDesire": "$node1 causes desire for [MASK]", 
		"/r/IsA": "$node1 is a [MASK]",
		"/r/SymbolOf": "$node1 is a symbol of [MASK]",
		"/r/MadeOf": "$node1 can be made of [MASK]", 
		"/r/LocatedNear": "$node1 is often located near [MASK]",
		"/r/Desires": "$node1 desires [MASK]",
		"/r/AtLocation": "$node1 can be found at [MASK]",
		"/r/HasProperty": "$node1 has property [MASK]",
		"/r/PartOf": "$node1 is part of [MASK]",
		"/r/HasFirstSubevent": "$node1 starts by [MASK]",
		"/r/HasLastSubevent": "$node1 ends by [MASK]"
	}
	if r in t.keys():
		temp=Template(t[r])
		question=temp.substitute(node1=h)
		template=temp.substitute(node1='{}').replace('[MASK]', '{}')
		return question, template
	else:
		print('ERROR')

def generate_questions(qa_pairs, rel_tails, answer_heads, output_file, embeddings, sentence2id, sentences, strategy, limit=1000):
	q_id=0
	all_rels=[]
	all_min_probs=[]
	with open(output_file, 'w') as w:
		for pair, qa_data in tqdm(qa_pairs.items(), total=len(qa_pairs)):
			node1, rel=pair
			n1_labels=qa_data[0][-1]
			for qa in qa_data:
				q,a, n1_labels, template, head_label, sent_source,distractor_only =qa
				if distractor_only or a in head_label:
					continue
				q_or_a = None
				if args.strategy == 'adv-answer':
					q_or_a = 'a'
				elif args.strategy == 'adv-question':
					q_or_a = 'q'
				if q_or_a != None:
					distractors, min_prob=select_distractors_af(rel_tails[rel], 
																head_label, 
																answer_heads,
																a,
																rel,
																q,
																embeddings, 
																sentence2id, 
																sentences,
																q_or_a)
				else:
					distractors, min_prob=select_distractors_noaf(rel_tails[rel],
																head_label,
																answer_heads,
																a,
																rel)
				if distractors:
					all_min_probs.append(min_prob)
					q_entry=format_question(q, a, distractors, q_id, head_label, template, sent_source, rel)
					q_id+=1
					w.write(json.dumps(q_entry) + '\n')
					all_rels.append(rel)
	print(Counter(all_rels))
	print(Counter(all_min_probs))

def get_labels(data):
	if '|' in data:
		return data.split('|')
	else:
		return [data]

def question_to_sentence(q):
	return q.replace('[MASK]', '').strip()

def make_masked_question(s):
	node1_start=s.find('[[')
	node1_end=s.find(']]')
	node1_label=s[node1_start+2:node1_end]

	node2_start=s.rfind('[[')
	node2_end=s.rfind(']]')
	new_s=s[:node2_start].replace('[[', '').replace(']]', '') + '[MASK]' + s[node2_end+2:]

	template=s[:node1_start] + '{}' + s[node1_end+2:node2_start] + '{}' + s[node2_end+2:]

	return new_s, node1_label, template

def make_masked_question_from_lex(sentence, head, tail):
	question=sentence.replace(tail, '[MASK]')
	template=sentence.replace(head, '{}').replace(tail, '{}')
	return question, template

def token_overlap(x, y):
	return bool(set(x.split()) & set(y.split()))

def build_embeddings(sentences, out_dir, model_name='roberta-large-nli-stsb-mean-tokens'):
	emb_file = os.path.join(args.out_dir, 'cwwv_emb.pkl')
	if path.exists(emb_file):
		print ('embeddings already exists, skip computation')
		with open(emb_file, 'rb') as f:
			data=pkl.load(f)
			embeddings=data['embeddings']
			sentences=data['sentences']
			return embeddings
	model = SentenceTransformer(model_name)
	embeddings = model.encode(sentences, show_progress_bar=True, device=0, num_workers=4)
	with open(emb_file, "wb") as fout:
		pkl.dump({'sentences': sentences, 'embeddings': embeddings}, fout, protocol=pkl.HIGHEST_PROTOCOL)
	return embeddings

def create_indices(cskg_file, lex_cache):
	qa_pairs=defaultdict(list)

	rel_tails=defaultdict(set)
	answer_heads=defaultdict(set)

	all_tails=set()

	q_sents=set()

	with open(cskg_file, 'r') as f:
		header=next(f)
		for line in f:
			fields=line.split('\t')

			# extract existing info
			node1=fields[1]
			rel=fields[2]
			node2=fields[3]
			pair=(node1, rel)
			node1_labels=get_labels(fields[4])
			#head_tokens=set()
			#for n1_label in node1_labels:
			#	head_tokens |= set(n1_label.split())
			node2_labels=get_labels(fields[5])
			edge_id=fields[0]
			source=fields[8]
			sentence=fields[9].strip()

			if '|' in source:
				source=set(source.split('|'))
			else:
				source=set([source])

			if rel not in good_relations or (len(source & (q_sources|dist_only_sources))==0): continue

			for answer in node2_labels:
				rel_tails[rel].add(answer)
				answer_heads[(answer, rel)] |= set(node1_labels)
				all_tails.add(answer)

			distractor_only=True
			for s in source:
				if s in q_sources:
					distractor_only=False
			if sentence:
				question, head_label, template = make_masked_question(sentence)
				for answer in node2_labels:
					if not token_overlap(head_label, answer):
						qa_pairs[pair].append((question, answer, node1_labels, template, head_label, 'omcs', distractor_only))
						q_sents.add(question_to_sentence(question))
			elif lex_cache:
				for n1_label in node1_labels:
					for answer in node2_labels:
						triple=(n1_label, rel, answer)
						if triple in lex_cache.keys() and not token_overlap(n1_label, answer):
							sentence=lex_cache[triple]
							question, template = make_masked_question_from_lex(sentence, n1_label, answer)
							if '[MASK]' not in question or template.split().count('{}')!=2 or question.split().count('[MASK]')==1:
								question, template = construct_from_template(n1_label, rel)
							qa_pairs[pair].append((question, answer, node1_labels, template, n1_label, 'lex', distractor_only))
							q_sents.add(question_to_sentence(question))
						elif not token_overlap(n1_label, answer):
							question, template = construct_from_template(n1_label, rel)
							qa_pairs[pair].append((question, answer, node1_labels, template, n1_label, 'lex', distractor_only))
							q_sents.add(question_to_sentence(question))
	return qa_pairs, all_tails, rel_tails, answer_heads, list(q_sents)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--cskg_file", type=str, default=None, required=True,
						help="CSKG graph TSV file")
	parser.add_argument("--out_dir", type=str, default=None, required=True,
						help="Output directory")
	parser.add_argument("--limit", type=int, default=1000000000,
						help="Limit of CSKG rows to process")
	parser.add_argument('--lex_cache', type=str, default='../cache.pkl',
						help="Pickle file that contains the cache of the lexicalization.")
	parser.add_argument("--strategy", default='random', type=str, required=False, choices=['random', 'adv-answer', 'adv-question'], help="which data generation strategy to use")
	args = parser.parse_args()
	
	lex_cache=None
	lex_cache=pkl.load(open(args.lex_cache, 'rb'))
	qa_pairs, all_tails, rel_tails, answer_heads, q_sentences=create_indices(args.cskg_file, lex_cache)
	print('Collecting sentences')
	sentences=list(all_tails) + q_sentences
	print(len(sentences), 'sentences', len(all_tails), 'answers', len(qa_pairs.keys()), 'qa pairs')
	if args.strategy == 'adv-answer' or args.strategy == 'adv-question':
		print ('Using %s strategy' % args.strategy)
		print('Computing embeddings')
		embeddings=build_embeddings(sentences, args.out_dir)
		print(len(embeddings), 'embeddings')
	else:
		embeddings = None
	sentence2id={word:i for i, word in enumerate(sentences)}
	output_file = path.join(args.out_dir, args.strategy+'.jsonl')
	generate_questions(qa_pairs, rel_tails, answer_heads, output_file, embeddings, sentence2id, sentences, args.strategy, args.limit)
	
		
