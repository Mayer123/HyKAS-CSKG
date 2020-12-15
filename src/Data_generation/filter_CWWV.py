from wordfreq import word_frequency
import json
from tqdm import tqdm
import argparse
import random 
import os
random.seed(1)
threshold=1e-06

def write_data(data, dest):
	with open(dest, 'w') as w:
		for x in data:
			w.write(json.dumps(x) + '\n')

def get_answer(data):
	answers={}
	for choice in data['question']['choices']:
		answers[choice['label']]=choice['text']
	return answers[data['answerKey']]


if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--input_file", type=str, default=None, required=True,
						help="Input file with artificial QA data")
	parser.add_argument('--do_split', action="store_true", help="Further split training set into subsets for AFLite")
	args = parser.parse_args()

	common_concepts_omcs=[]
	with open(args.input_file, 'r') as f:
		for line in tqdm(f, total=500000):
			qdata=json.loads(line)
			head_label=qdata['question']['head']
			source=qdata['question']['source']
			answer=get_answer(qdata)
			is_concept=source=='omcs' or (head_label.islower() and answer.islower())
			is_common=(word_frequency(head_label, 'en')>=threshold and word_frequency(answer, 'en')>=threshold)
			if is_concept and is_common and source == 'omcs':
				common_concepts_omcs.append(qdata)

	print('common concepts omcs', len(common_concepts_omcs))
	random.shuffle(common_concepts_omcs)
	train_set = common_concepts_omcs[:int(len(common_concepts_omcs)*0.95)]
	dev_set = common_concepts_omcs[int(len(common_concepts_omcs)*0.95):]
	basename = os.path.basename(args.input_file)
	write_data(train_set, args.input_file.replace(basename, 'train_'+basename))
	write_data(dev_set, args.input_file.replace(basename, 'dev_'+basename))
	if args.do_split:
		assert 'random' in args.input_file
		print ('splitting train into subsets, which can be used for AFLite (only valid for random strategy)')
		train_set_1 = train_set[:int(len(train_set)*0.01)]
		train_set_4 = train_set[int(len(train_set)*0.01):int(len(train_set)*0.05)]
		train_set_95 = train_set[int(len(train_set)*0.05):]
		write_data(train_set_1, 'train_1%_'+args.input_file)
		write_data(train_set_4, 'train_4%_'+args.input_file)
		write_data(train_set_95, 'train_95%_'+args.input_file)
