# Knowledge-driven Data Construction for Zero-shot Evaluation in Commonsense Question Answering
This repository contains the code for the paper "Knowledge-driven Data Construction for Zero-shot Evaluation in Commonsense Question Answering" (AAAI 2021). See full paper [here](https://arxiv.org/abs/2011.03863)

Note that our evaluation code is adpated from [self-talk repo](https://github.com/vered1986/self_talk)

## Enviroments
This code has been tested on Python 3.7.6, Pytorch 1.5.1 and Transformers 3.0.2, you can install the required packages by 
```
pip install -r requirements.txt
```

## Data generation
Our synthetic QA sets can be downloaded from [here](https://drive.google.com/file/d/1qp2Exh88m1LT8iyDvt8TOAXhGdHQhP2B/view?usp=sharing), uncompress it and place it in the HyKAS-CSKG root directory.

If you would like to generate data from scratch, first `cd` to the `src/Data_generation` directory.

For the **ATOMIC** synthetic sets, download the ATOMIC from [official website](https://homes.cs.washington.edu/~msap/atomic/) and uncompress.
Then run
```
python generate_from_ATOMIC.py --train_KG atomic/v4_atomic_trn.csv --dev_KG atomic/v4_atomic_dev.csv --strategy random --out_dir ../../data/ATOMIC  
```

For **CWWV**, download the `cskg_connected.tsv` from [here](https://drive.google.com/file/d/11TiW3pAHnt6l8yuIWpowzOMuM8fq7ff6/view?usp=sharing) and `cache.pkl` from [here](https://drive.google.com/file/d/19tcSaKi-Efz8IH-HX0oBkYtalnqOseZj/view?usp=sharing), then run:
```
python generate_from_CWWV.py --cskg_file cskg_connected.tsv --lex_cache cache.pkl --output_dir ../../data/CWWV --strategy random
python filter_CWWV.py --input_file ../../data/CWWV/random.jsonl 
```

## Pretraining on Synthetic QA sets
We provide following pretrained models 
LM | KG | Download
---|---|---
RoBERTa-Large | ATOMIC | [Download](https://drive.google.com/file/d/1oTYV5YZRlXtMSZW9_pTjyMn6o8yrPU2N/view?usp=sharing)
RoBERTa-Large | CWWV | [Download](https://drive.google.com/file/d/1Ot-x3WJoFWYUTyyDSMeG2CrKDmCTggxM/view?usp=sharing)
RoBERTa-Large | CSKG | [Download](https://drive.google.com/file/d/1nfWtIfrQk4REp7oGUyyn1ShT7aEvMI9E/view?usp=sharing)
GPT2-Large | ATOMIC | [Download](https://drive.google.com/file/d/1lENyTTBogmRIK_M7cu_uxeD8AiWBo7Ko/view?usp=sharing)
GPT2-Large | CWWV | [Download](https://drive.google.com/file/d/1dnqdW-5d6tULZfDaejViVrjuNx-nY8sP/view?usp=sharing)
GPT2-Large | CSKG | [Download](https://drive.google.com/file/d/1VUBAxtyKElmbNTxSkIdPjR88PkEjbc-2/view?usp=sharing)

If you would like to train models from scratch, you can use the following commands under src/Training

For RoBERTa
```
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --model_type roberta-mlm --model_name_or_path roberta-large --task_name cskg --output_dir ../../out_dir --max_sequence_per_time 200 \
--train_file ../../data/ATOMIC/train_random.jsonl --second_train_file ../../data/CWWV/train_random.jsonl --dev_file ../../data/ATOMIC/dev_random.jsonl --second_dev_file \
../../data/CWWV/dev_random.jsonl --max_seq_length 128 --max_words_to_mask 6 --do_train --do_eval --per_gpu_train_batch_size 2 --gradient_accumulation_steps 16 \
--learning_rate 1e-5 --num_train_epochs 1 --warmup_proportion 0.05 --evaluate_during_training --per_gpu_eval_batch_size 8  --save_steps 6500 --margin 1.0
```
For GPT2 
```
CUDA_VISIBLE_DEVICES=0 python run_pretrain_gpt2.py --model_type gpt2 --model_name_or_path gpt2-large --task_name cskg --output_dir ../../out_dir --train_file ../../data/ATOMIC/ \
train_random.jsonl --second_train_file ../../data/CWWV/train_random.jsonl --dev_file ../../data/ATOMIC/dev_random.jsonl --second_dev_file ../../data/CWWV/dev_random.jsonl \
--max_seq_length 128 --do_train --do_eval --per_gpu_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 1e-5 --num_train_epochs 1 --warmup_proportion 0.05 \
--evaluate_during_training --per_gpu_eval_batch_size 8  --save_steps 6500 --margin 1.0
```

## Evaluation
For LM baselines, cd to src/Evaluation directory
```
python evaluate_RoBERTa.py --lm roberta-large --dataset_file DATA_FILE --out_dir ../../results --device 1 --reader TASK_NAME
python evaluate_GPT2.py --lm gpt2-large --dataset_file DATA_FILE --out_dir ../../results --device 1  --reader TASK_NAME
```
For pretrained models, simply point the --lm flag to your model directory, for example 
```
python evaluate_RoBERTa.py --lm ../../models/roberta_cskg --dataset_file ../../tasks/commonsenseqa_dev.jsonl --out_dir ../../results --device 1 --reader commonsenseqa
python evaluate_GPT2.py --lm ../../models/gpt2_cskg --dataset_file ../../tasks/commonsenseqa_dev.jsonl --out_dir ../../results --device 1  --reader commonsenseqa
```

## MLM abalation
To run MLM pretraining experiments (comparison of training regimes), cd to src/Training/MLM 
```
CUDA_VISIBLE_DEVICES=0 python run_mlm_roberta.py --model_type roberta-mlm --model_name_or_path roberta-large --task_name atomicmlm --output_dir ../../out_dir --train_file \
../../data/ATOMIC/train_random.jsonl --dev_file ../../data/ATOMIC/dev_random.jsonl --mlm_probability 0.5 --max_seq_length 128 --max_words_to_mask 6 --max_sequence_per_time 200 \
--do_train --do_eval --per_gpu_train_batch_size 8 --gradient_accumulation_steps 4 --learning_rate 1e-5 --num_train_epochs 3 --warmup_proportion 0.05 --evaluate_during_training \
--per_gpu_eval_batch_size 8 --save_steps 5000
```
Then follow the same evaluation commands as above to evaluate the models

## AFLite
To generate adversarial filtered datasets using AFLite algorithm, first run the data generation code with --do_split flag 
```
python generate_from_ATOMIC.py --train_KG atomic/v4_atomic_trn.csv --dev_KG atomic/v4_atomic_dev.csv --strategy random --out_dir ../../data/ATOMIC --do_split 
```
This will split training set into 3 subsets, then we can train a feature function, cd to src/Training/AFLite directory
```
CUDA_VISIBLE_DEVICES=0 python run_roberta_classification.py --model_type roberta-mc --model_name_or_path roberta-large --task_name cwwv --output_dir ../../out_dir --train_file \
../../data/ATOMIC/train_4%_random.jsonl  --dev_file ../../data/ATOMIC/train_1%_random.jsonl --max_seq_length 128 --per_gpu_eval_batch_size 16 --do_train --do_eval \
--evaluate_during_training --per_gpu_train_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-5 --num_train_epochs 3 --warmup_proportion 0.05 --save_steps 150 
```
Then we compute the embeddings for the remaining 95% of train and dev sets 
```
CUDA_VISIBLE_DEVICES=0 python run_roberta_classification.py --model_type roberta-mc --model_name_or_path roberta-large --task_name cwwv --output_dir ../../out_dir --train_file \
../../data/ATOMIC/train_4%_random.jsonl  --dev_file ../../data/ATOMIC/train_95%_random.jsonl --max_seq_length 128 --per_gpu_eval_batch_size 16 --do_eval
CUDA_VISIBLE_DEVICES=0 python run_roberta_classification.py --model_type roberta-mc --model_name_or_path roberta-large --task_name cwwv --output_dir ../../out_dir --train_file \
../../data/ATOMIC/train_4%_random.jsonl  --dev_file ../../data/ATOMIC/dev_random.jsonl --max_seq_length 128 --per_gpu_eval_batch_size 16 --do_eval
```
To run AFLite 
```
python run_AFLite.py --train_file ../../data/ATOMIC/train_95%_random.jsonl  --dev_file ../../data/ATOMIC/dev_random.jsonl 
```
This will produce the AFLite filtered output files at the same location as input files, which can be used for pretraining the models. 

## Cite 
```
@misc{ma2020knowledgedriven,
    title={Knowledge-driven Data Construction for Zero-shot Evaluation in Commonsense Question Answering},
    author={Kaixin Ma and Filip Ilievski and Jonathan Francis and Yonatan Bisk and Eric Nyberg and Alessandro Oltramari},
    year={2020},
    eprint={2011.03863},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
