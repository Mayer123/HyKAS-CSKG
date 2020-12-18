# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import sys
sys.path.append('../')
sys.path.append('.')
from transformers import (WEIGHTS_NAME, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from transformers import AdamW, get_linear_schedule_with_warmup
from data_utils import myprocessors, handle_underscores
import json
from collections import Counter
logger = logging.getLogger(__name__)
from transformers import MODEL_WITH_LM_HEAD_MAPPING
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

MODEL_CLASSES = {
	'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}

class MyDataset(torch.utils.data.Dataset):
	def __init__(self, data, mask_token):
		self.data = data
		self.mask_token = mask_token

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		return sample, self.mask_token

def mCollateFn(batch):
	batch_input_ids = []
	batch_input_mask = []
	batch_label_ids = []
	mask_token = batch[0][1]
	max_len = max([len(f[0]) for f in batch])
	for f in batch:
		input_ids = np.full(max_len, mask_token)
		input_ids[:len(f[0])] = f[0]
		labels = np.array([-100 if f[0][i] == mask_token else f[0][i] for i in range(len(f[0]))]+[-100]*(max_len-len(f[0])))
		mask = np.zeros(max_len)
		mask[:len(f[0])] = 1
		batch_input_ids.append(input_ids)
		batch_input_mask.append(mask)
		batch_label_ids.append(labels)
		
	batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
	batch_input_mask = torch.tensor(batch_input_mask, dtype=torch.long)
	batch_label_ids = torch.tensor(batch_label_ids, dtype=torch.long)
	return batch_input_ids, batch_input_mask, batch_label_ids

def convert_examples_to_features(examples, tokenizer, max_length=512):
	data = []
	for example in examples:
		inputs, _ = handle_underscores(example['context'], tokenizer, keywords=example['keywords'], prefix=True)
		t_inputs, _ = handle_underscores(example['ending'], tokenizer)
		input_ids = tokenizer.convert_tokens_to_ids(inputs+t_inputs)
		data.append(input_ids)
	return data

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args, train_dataset, model, tokenizer):
	""" Train the model """
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=mCollateFn)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]

	warmup_steps = args.warmup_steps if args.warmup_steps != 0 else int(args.warmup_proportion * t_total)
	logger.info("warm up steps = %d", warmup_steps)
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(0.9, 0.98))
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

	# multi-gpu training (should be after apex fp16 initialization)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
														  output_device=args.local_rank,
														  find_unused_parameters=True)
	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
				   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
	set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		for step, batch in enumerate(epoch_iterator):
			model.train()
			inputs = {'input_ids':      batch[0].cuda(),
					  'attention_mask': batch[1].cuda(),
					  'labels': batch[2].cuda()}
			outputs = model(**inputs)
			loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

			if args.n_gpu > 1:
				loss = loss.mean() # mean() to average on multi-gpu parallel training
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			if args.fp16:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1

				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
					tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
					tb_writer.add_scalar('Batch_loss', loss.item()*args.gradient_accumulation_steps, global_step)
					logger.info(" global_step = %s, average loss = %s", global_step, (tr_loss - logging_loss)/args.logging_steps)
					logging_loss = tr_loss
					

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break
		if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
			# Save model checkpoint
			output_dir = os.path.join(args.output_dir, 'epoch%s' % _)
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)
			model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
			model_to_save.save_pretrained(output_dir)
			tokenizer.save_pretrained(output_dir)
			torch.save(args, os.path.join(output_dir, 'training_args.bin'))
			logger.info("Saving model checkpoint to %s", output_dir)
	if args.local_rank in [-1, 0]:
		tb_writer.close()
	return global_step, tr_loss / global_step


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
	if args.local_rank not in [-1, 0] and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

	processor = myprocessors[task](args)
	examples = processor.get_train_examples()
	features = convert_examples_to_features(examples, tokenizer, max_length=args.max_seq_length)
	if args.local_rank == 0 and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
	return MyDataset(features, tokenizer.mask_token_id)

def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--train_file", default=None, type=str, required=True,
						help="The train file name")
	parser.add_argument("--dev_file", default=None, type=str, required=True,
						help="The dev file name")
	parser.add_argument("--model_type", default=None, type=str, required=True,
						help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
	parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
						help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_TYPES))
	parser.add_argument("--config_name", default="", type=str,
						help="Pretrained config name or path if not the same as model_name")
	parser.add_argument("--tokenizer_name", default="", type=str,
						help="Pretrained tokenizer name or path if not the same as model_name")
	parser.add_argument("--cache_dir", default="", type=str,
						help="Where do you want to store the pre-trained models downloaded from s3")
	parser.add_argument("--task_name", default=None, type=str, required=True,
						help="The name of the task to train selected in the list: " + ", ".join(myprocessors.keys()))
	parser.add_argument("--output_dir", default=None, type=str, required=True,
						help="The output directory where the model predictions and checkpoints will be written.")

	## Other parameters
	parser.add_argument("--max_seq_length", default=128, type=int,
						help="The maximum total input sequence length after tokenization. Sequences longer "
							 "than this will be truncated, sequences shorter will be padded.")
	parser.add_argument("--do_train", action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval", action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--do_lower_case", action='store_true',
						help="Set this flag if you are using an uncased model.")
	parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
						help="Batch size per GPU/CPU for training.")
	parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
						help="Batch size per GPU/CPU for evaluation.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--learning_rate", default=1e-5, type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.01, type=float,
						help="Weight deay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-6, type=float,
						help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")
	parser.add_argument("--num_train_epochs", default=1.0, type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--max_steps", default=-1, type=int,
						help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
	parser.add_argument("--warmup_steps", default=0, type=int,
						help="Linear warmup over warmup_steps.")
	parser.add_argument("--warmup_proportion", default=0.05, type=float,
						help="Linear warmup over warmup proportion.")
	parser.add_argument('--logging_steps', type=int, default=50,
						help="Log every X updates steps.")
	parser.add_argument('--save_steps', type=int, default=50,
						help="Save checkpoint every X updates steps.")
	parser.add_argument("--no_cuda", action='store_true',
						help="Avoid using CUDA when available")
	parser.add_argument('--overwrite_output_dir', action='store_true',
						help="Overwrite the content of the output directory")
	parser.add_argument('--seed', type=int, default=2555,
						help="random seed for initialization")
	parser.add_argument('--fp16', action='store_true',
						help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
	parser.add_argument('--fp16_opt_level', type=str, default='O1',
						help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
							 "See details at https://nvidia.github.io/apex/amp.html")
	parser.add_argument("--local_rank", type=int, default=-1,
						help="For distributed training: local_rank")
	parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
	parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
	args = parser.parse_args()

	if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
		raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend='nccl')
		args.n_gpu = 1
	args.device = device

	if args.do_train:
		for handler in logging.root.handlers[:]:
			logging.root.removeHandler(handler)
	# Setup logging
	if args.do_train:
		log_file = os.path.join(args.output_dir, 'train.log')
		logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
							datefmt = '%m/%d/%Y %H:%M:%S',
							level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
							filename=log_file)
		logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
						args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
		os.system("cp run_lm_gpt2.py %s" % os.path.join(args.output_dir, 'run_lm_gpt2.py'))
		os.system("cp ../data_utils.py %s" % os.path.join(args.output_dir, 'data_utils.py'))

	# Set seed
	set_seed(args)
	args.task_name = args.task_name.lower()
	if args.task_name not in myprocessors:
		raise ValueError("Task not found: %s" % (args.task_name))
	
	args.model_type = args.model_type.lower()
	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, finetuning_task=args.task_name, cache_dir=args.cache_dir)
	tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
	model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir)
	
	count = count_parameters(model)
	print (count)
	special_tokens_dict = {'mask_token': '<mask>'}
	num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
	model.resize_token_embeddings(len(tokenizer))

	if args.local_rank == 0:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	model.to(args.device)

	logger.info("Training/evaluation parameters %s", args)
	if args.do_train:
		train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
		global_step, tr_loss = train(args, train_dataset, model, tokenizer)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
	# Evaluation
	results = {}

	return results

if __name__ == "__main__":
	main()