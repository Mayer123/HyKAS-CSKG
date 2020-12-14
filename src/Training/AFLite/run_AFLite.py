import json
import random
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm
import argparse
correct_count = Counter()
chosen_count = Counter()
dev_correct_count = Counter()
dev_chosen_count = Counter()

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

def read_data(filename):
	data = []
	with open(filename, 'r') as f:
		for line in f:
			data.append(json.loads(line))
	return data

def write_data(filename, data):
	with open(filename, 'w') as fout:
		for sample in data:
			fout.write(json.dumps(sample))
			fout.write('\n')

def predict(model, features, labels):
	n_samples, num_cand, feat_dim = features.shape
	if len(features) > 500000:
		logits = []
		batch_size = int(len(features)/10) 
		for b in range(0, len(features), batch_size):
			batch_logits =  model(features[b:b+batch_size].cuda())
			logits.append(batch_logits.squeeze(2).detach().cpu())
		logits = torch.cat(logits, dim=0).numpy()
	else:
		features = features.cuda()
		logits = model(features)	
		logits = logits.squeeze(2).detach().cpu().numpy()
	preds = np.argmax(logits, axis=1)
	acc = (preds == labels).mean()
	return preds == labels

def train_classifier(features, labels):
	model = torch.nn.Linear(1024, 1)
	model.to('cuda')
	optimizer = torch.optim.Adam(model.parameters())
	loss_fct = torch.nn.CrossEntropyLoss()
	features = features.cuda()
	labels = torch.tensor(labels, dtype=torch.long).cuda()
	batch_size = int(len(features)/10) 
	for i in range(3):

		for b in range(0, len(features), batch_size):
			logits = model(features[b:b+batch_size])
			loss = loss_fct(logits.squeeze(2), labels[b:b+batch_size])
			loss.backward()
			optimizer.step()
			model.zero_grad()
	return model

def run_iteration(features, labels, sample_ids, test_features, test_labels, test_sample_ids, target_size):
	global correct_count, chosen_count, dev_correct_count, dev_chosen_count
	idx = [_ for _ in range(len(features))]
	random.shuffle(idx)
	features = features[idx]
	labels = labels[idx]
	sample_ids = [sample_ids[i] for i in idx]
	train_size = target_size
	train_feat = features[:train_size]
	dev_feat = features[train_size:]
	train_labels = labels[:train_size]
	dev_labels = labels[train_size:]
	train_sample_ids = sample_ids[:train_size]
	dev_sample_ids = sample_ids[train_size:]
	model = train_classifier(train_feat, train_labels)
	preds = predict(model, dev_feat, dev_labels)
	chosen_count.update(dev_sample_ids)
	correct_ids = [dev_sample_ids[sid] for sid in range(len(dev_sample_ids)) if preds[sid]]
	correct_count.update(correct_ids)
	test_preds = predict(model, test_features, test_labels)
	dev_chosen_count.update(test_sample_ids)
	test_correct_ids = [test_sample_ids[sid] for sid in range(len(test_sample_ids)) if test_preds[sid]]
	dev_correct_count.update(test_correct_ids)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_file", default=None, type=str, required=True, help="train file")
	parser.add_argument("--dev_file", default=None, type=str, required=True, help="dev file")
	args = parser.parse_args()
	set_seed(1)
	data = read_data(args.train_file)
	dev_data = read_data(args.dev_file)
	print (len(data), len(dev_data))
	features = torch.load(args.train_file.replace('.jsonl', '_features'))
	torch_labels = torch.load(args.train_file.replace('.jsonl', '_labels'))
	dev_features = torch.load(args.dev_file.replace('.jsonl', '_features'))
	dev_torch_labels = torch.load(args.dev_file.replace('.jsonl', '_labels'))
	print (features.shape, dev_features.shape)
	if 'correct' in data[0]:
		labels = [sample['correct'] for sample in data]
		dev_labels = [sample['correct'] for sample in dev_data]
	else:
		mapping = {'A':0, 'B':1, 'C':2}
		labels = [mapping[sample['answerKey']] for sample in data]
		dev_labels = [mapping[sample['answerKey']] for sample in dev_data]
	print (torch_labels.shape, dev_torch_labels.shape)
	print (np.array(labels).shape, np.array(dev_labels).shape)
	assert all(np.array(labels) == torch_labels)
	assert all(np.array(dev_labels) == dev_torch_labels)
	sample_ids = [sample['id'] for sample in data]
	dev_sample_ids = [sample['id'] for sample in dev_data]
	labels = np.array(labels)
	dev_labels = np.array(dev_labels)
	target_size = int(len(features)*0.2)
	cutoff_size = int(len(features)*0.02)
	dev_cutoff_size = int(len(dev_features)*0.02)
	print ('target size', target_size)
	global correct_count, chosen_count, dev_correct_count, dev_chosen_count
	while len(features) > target_size:
		correct_count = Counter()
		chosen_count = Counter()
		dev_correct_count = Counter()
		dev_chosen_count = Counter()
		for i in tqdm(range(64)):
			run_iteration(features, labels, sample_ids, dev_features, dev_labels, dev_sample_ids, target_size)
		for k, v in correct_count.items():
			correct_count[k] = float(v)/chosen_count[k]
		for k, v in dev_correct_count.items():
			dev_correct_count[k] = float(v)/dev_chosen_count[k]
		sorted_correct_count = sorted(correct_count.items(), key=lambda x: x[1], reverse=True)
		sorted_dev_correct_count = sorted(dev_correct_count.items(), key=lambda x: x[1], reverse=True)
		easy_train = [s[0] for s in sorted_correct_count[:cutoff_size] if s[1] > 0.75]
		easy_dev = [s[0] for s in sorted_dev_correct_count[:dev_cutoff_size] if s[1] > 0.75]

		kept_idx = [sid for sid in range(len(sample_ids)) if sample_ids[sid] not in easy_train]
		newly_removed = len(features) - len(kept_idx)
		features = features[kept_idx]
		labels = labels[kept_idx]
		sample_ids = [sample_ids[ki] for ki in kept_idx]
		dev_kept_ids = [sid for sid in range(len(dev_sample_ids)) if dev_sample_ids[sid] not in easy_dev]
		dev_features = dev_features[dev_kept_ids]
		dev_labels = dev_labels[dev_kept_ids]
		dev_sample_ids = [dev_sample_ids[ki] for ki in dev_kept_ids]
		print ('now keeping train', len(kept_idx), 'dev', len(dev_kept_ids))
		if newly_removed < cutoff_size:
			break
	print ('finally keeping train', len(sample_ids), 'dev', len(dev_sample_ids))
	kept = Counter(sample_ids)
	kept_data = [sample for sample in data if sample['id'] in kept]
	dev_kept = Counter(dev_sample_ids)
	dev_kept_data = [sample for sample in dev_data if sample['id'] in dev_kept]
	write_data(args.train_file.replace('.jsonl', '_adv-filter.jsonl'), kept_data)
	write_data(args.dev_file.replace('.jsonl', '_adv-filter.jsonl'), dev_kept_data)

if __name__ == "__main__":
	main()