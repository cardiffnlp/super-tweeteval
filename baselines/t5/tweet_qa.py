"""
python tweet_qa.py -m 'google/flan-t5-small' --model-alias 'flan-t5-small-tweetqa' --n-trials 20 \
--use-auth-token --model-organization cardiffnlp
python tweet_qa.py -m 'google/flan-t5-base' --model-alias 'flan-t5-base-tweetqa' --n-trials 20 \
--use-auth-token --model-organization cardiffnlp
python tweet_qa.py -m 'google/flan-t5-small' --model-alias 'flan-t5-small-tweetqa' --n-trials 20 \
--use-auth-token --model-organization cardiffnlp
python tweet_qa.py -m 'google/flan-t5-base' --model-alias 'flan-t5-base-tweetqa' --n-trials 20 \
--use-auth-token --model-organization cardiffnlp
"""
import json
import logging
import os
import urllib
import multiprocessing
import argparse
import gc
from typing import List

import torch
import transformers
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline
from ray import tune, init
from evaluate import load
from huggingface_hub import Repository


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # turn-off the warning message
os.environ["WANDB_DISABLED"] = "true"
local_files_only = True
try:
    urllib.request.urlopen('http://google.com')
except:
    local_files_only = False


def load_model(model_name: str, cache_dir: str = None, use_auth_token: bool = False, low_cpu_mem_usage: bool = False):
    """ load language model from huggingface model hub """
    # config & tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_name, local_files_only=local_files_only, cache_dir=cache_dir, use_auth_token=use_auth_token)
    if config.model_type == 't5':  # T5 model requires T5ForConditionalGeneration class
        model_class = transformers.T5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'mt5':
        model_class = transformers.MT5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'bart':
        model_class = transformers.BartForConditionalGeneration.from_pretrained
    elif config.model_type == 'mbart':
        model_class = transformers.MBartForConditionalGeneration.from_pretrained
    else:
        raise ValueError(f'unsupported model type: {config.model_type}')
    param = {'config': config, "local_files_only": local_files_only, "use_auth_token": use_auth_token, "low_cpu_mem_usage": low_cpu_mem_usage, "cache_dir": cache_dir}
    model = model_class(model_name, **param)
    return model


def train(model_name: str, model_low_cpu_mem_usage: bool, dataset: str, dataset_name: str,
          dataset_column_question: str, dataset_column_passage: str, dataset_column_answer: str, dataset_split_train: str,
          dataset_split_validation: str, dataset_split_test: str, search_range_lr: List, search_range_epoch: List,
          search_list_batch: List, down_sample_train: int, down_sample_validation: int, random_seed: int,
          use_auth_token: bool, n_trials: int, eval_step: int, parallel_cpu: bool, cache_dir: str, output_dir: str,
          ray_result_dir: str, model_alias: str, model_organization: str):
    """ fine-tune seq2seq model on qa """
    logging.info(f'[CONFIG]\n\t *LM: {model_name}, \n\t *Data: {dataset} ({dataset_name}), \n\t *Num of Trial: {n_trials}')
    # set up the output directory
    if output_dir is None:
        output_dir = f'ckpt/{os.path.basename(model_name)}.{os.path.basename(dataset)}.{dataset_name}'
    ray_result_dir = ray_result_dir
    if ray_result_dir is None:
        ray_result_dir = f'ray/{os.path.basename(model_name)}.{os.path.basename(dataset)}.{dataset_name}'

    # define search space
    search_range_lr = [1e-6, 1e-4] if search_range_lr is None else search_range_lr
    assert len(search_range_lr) == 2, f"`search_range_lr` should contain [min_lr, max_lr]: {search_range_lr}"
    search_range_epoch = [2, 6] if search_range_epoch is None else search_range_epoch
    assert len(search_range_epoch) == 2, f"`search_range_epoch` should contain [min_epoch, max_epoch]: {search_range_epoch}"
    search_list_batch = [64, 128] if search_list_batch is None else search_list_batch
    search_space = {
        "learning_rate": tune.loguniform(search_range_lr[0], search_range_lr[1]),
        "num_train_epochs": tune.choice(list(range(search_range_epoch[0], search_range_epoch[1]))),
        "per_device_train_batch_size": tune.choice(search_list_batch)
    }
    resources_per_trial = {'cpu': multiprocessing.cpu_count() if parallel_cpu else 1, "gpu": torch.cuda.device_count()}
    init(ignore_reinit_error=True, num_cpus=resources_per_trial['cpu'])
    logging.info(f'[RESOURCE]\n{json.dumps(resources_per_trial, indent=4)}')

    # dataset process
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=local_files_only, use_auth_token=use_auth_token)
    dataset_split = {
        "train": [dataset_split_train, down_sample_train],
        "validation": [dataset_split_validation, down_sample_validation]
    }
    dataset_instance = load_dataset(dataset, dataset_name, use_auth_token=use_auth_token)
    tokenized_dataset = {}
    for s, (s_dataset, down_sample) in dataset_split.items():
        tokenized_dataset[s] = []
        tmp = dataset_instance[s_dataset]
        tmp.shuffle(random_seed)
        for i in tmp:
            model_inputs = tokenizer(f"context: {i[dataset_column_passage]}, question: {i[dataset_column_question]}", truncation=True)
            model_inputs['labels'] = tokenizer(text_target=i[dataset_column_answer], truncation=True)['input_ids']
            tokenized_dataset[s].append(model_inputs)

        if down_sample is not None and len(tmp) > down_sample:
            tokenized_dataset[f"{s}_ds"] = []
            tmp = tmp.select(list(range(down_sample)))
            for i in tmp:
                model_inputs = tokenizer(f"context: {i[dataset_column_passage]}, question: {i[dataset_column_question]}", truncation=True)
                model_inputs['labels'] = tokenizer(text_target=i[dataset_column_answer], truncation=True)['input_ids']
                tokenized_dataset[f"{s}_ds"].append(model_inputs)
        else:
            tokenized_dataset[f"{s}_ds"] = tokenized_dataset[s]

    # metric
    metric = load("squad")

    def get_metric(target_metric: str = None):
        def compute_metric(eval_pred):  # for parameter search
            predictions, reference_token_ids = eval_pred
            # format reference
            references_decode = [tokenizer.decode(list(filter(lambda x: x != -100, r)), skip_special_tokens=True) for r in reference_token_ids]
            references = [{"answers": {"answer_start": [100], "text": [r]}, "id": str(_n)} for _n, r in enumerate(references_decode)]
            # format prediction
            logit, loss = predictions
            generation_token_id = logit.argmax(-1)
            generation_token_id[logit.min(-1) == -100] = -100
            generation_decode = [tokenizer.decode(list(filter(lambda x: x != -100, r)), skip_special_tokens=True) for r in generation_token_id]
            predictions = [{"prediction_text": p, "id": str(_n)} for _n, p in enumerate(generation_decode)]
            if target_metric is not None:
                return metric.compute(predictions=predictions, references=references)[target_metric]
            return metric.compute(predictions=predictions, references=references)

        return compute_metric

    if not os.path.exists(f"{output_dir}/model/pytorch_model.bin"):
        trainer = Seq2SeqTrainer(
            # model=model,
            args=Seq2SeqTrainingArguments(
                output_dir=f"{output_dir}/runs",
                evaluation_strategy="steps",
                eval_steps=eval_step,
                seed=random_seed
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=load_model(
                model_name=model_name,
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
                low_cpu_mem_usage=model_low_cpu_mem_usage)),
            train_dataset=tokenized_dataset['train_ds'],
            eval_dataset=tokenized_dataset['validation_ds'],
            compute_metrics=get_metric("f1"),
            model_init=lambda x: load_model(
                model_name=model_name,
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
                low_cpu_mem_usage=model_low_cpu_mem_usage)
        )
        os.makedirs(f"{output_dir}/model", exist_ok=True)
        if not os.path.exists(f"{output_dir}/model/hyperparameters.json"):
            # grid search
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: search_space,
                local_dir=ray_result_dir,
                direction="maximize",
                backend="ray",
                n_trials=n_trials,
                resources_per_trial=resources_per_trial
            )
            with open(f"{output_dir}/model/hyperparameters.json", 'w') as f:
                json.dump(best_run.hyperparameters, f)
        else:
            logging.info("skip hyperparameter search (already done)")

        # fine-tuning with the best config
        logging.info(f"fine-tuning with the best config")
        with open(f"{output_dir}/model/hyperparameters.json") as f:
            best_hyperparameters = json.load(f)
        for n, v in best_hyperparameters.items():
            setattr(trainer.args, n, v)
        setattr(trainer, "train_dataset", tokenized_dataset['train'])
        setattr(trainer.args, "evaluation_strategy", 'no')
        trainer.train()
        trainer.save_model(f"{output_dir}/model")
        tokenizer.save_pretrained(f"{output_dir}/model")
        logging.info(f"model saved at {output_dir}/model")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logging.info("skip hyperparameter search & model training (already done)")

    # get metric on the test set
    if dataset_split_test is not None and not os.path.exists(f"{output_dir}/model/evaluation_metrics.json"):
        logging.info("run evaluation on test set")
        if not os.path.exists(f"{output_dir}/model/prediction_test.txt"):
            pipe = pipeline('text2text-generation', model=f"{output_dir}/model", device=1 if resources_per_trial['gpu'] > 0 else 0)
            input_data = [f"context: {i[dataset_column_passage]}, question: {i[dataset_column_question]}" for i in dataset_instance[dataset_split_test]]
            output = pipe(input_data)
            output = [i['generated_text'] for i in output]
            with open(f"{output_dir}/model/prediction_test.txt", "w") as f:
                f.write("\n".join(output))
        with open(f"{output_dir}/model/prediction_test.txt"):
            output = [i for i in f.read().split("\n") if len(i) > 0]
            predictions = [{"prediction_text": p, "id": str(_n)} for _n, p in enumerate(output)]
        references = [{"answers": {"answer_start": [100], "text": [r[dataset_column_answer]]}, "id": str(_n)} for _n, r in enumerate(dataset_instance[dataset_split_test])]
        eval_metric = metric.compute(predictions=predictions, references=references)
        logging.info(json.dumps(eval_metric, indent=4))
        with open(f"{output_dir}/model/evaluation_metrics.json", 'w') as f:
            json.dump(eval_metric, f)

    if model_alias is not None:
        assert model_organization is not None, "model_organization must be specified when model_alias is specified"
        logging.info('uploading to huggingface')
        args = {"use_auth_token": use_auth_token, "organization": model_organization}
        model = load_model(model_name=f"{output_dir}/model")
        model.push_to_hub(model_alias, **args)
        tokenizer.push_to_hub(model_alias, **args)
        # repo = Repository(model_alias, organization=model_organization)
        repo = Repository(model_alias, f"{model_organization}/{model_alias}")

        sample = [f"context: {i[dataset_column_passage]}, question: {i[dataset_column_question]}" for i in dataset_instance[dataset_split_train]][:3]
        widget = "\n".join([f"- text: {t}\n  example_title: example {_n + 1}" for _n, t in enumerate(sample)])
        with open(f"{output_dir}/model/README.md", "w") as f:
            f.write(f"""
---
widget:{widget}
---
# {model_organization}/{model_alias}

This is [{model_name}](https://huggingface.co/{model_name}) fine-tuned on [{dataset} ({dataset_name})](https://huggingface.co/datasets/{dataset}).

### Usage

```python
from transformers import pipeline

pipe = pipeline('text2text-generation', model="{model_organization}/{model_alias}")
output = pipe("{sample[0]}")
```
        """)
        repo.push_to_hub()


if __name__ == '__main__':
    # arguments
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description='Seq2Seq LM Fine-tuning on QA.')
    parser.add_argument('-m', '--model-name', default='google/flan-t5-small', type=str)
    parser.add_argument('--low-cpu-mem-usage', action='store_true')
    parser.add_argument('-d', '--dataset', default="cardiffnlp/super_tweeteval", type=str)
    parser.add_argument('--dataset-name', default="tweet_qa", type=str)
    parser.add_argument('--dataset-column-question', default="context", type=str)
    parser.add_argument('--dataset-column-passage', default="text", type=str)
    parser.add_argument('--dataset-column-answer', default="gold_label_str", type=str)
    parser.add_argument('--dataset-split-train', default="train", type=str)
    parser.add_argument('--dataset-split-validation', default="validation", type=str)
    parser.add_argument('--dataset-split-test', default="test", type=str)
    parser.add_argument('--search-range-lr', nargs='+', default=None, type=float)
    parser.add_argument('--search-range-epoch', nargs='+', default=None, type=int)
    parser.add_argument('--search-list-batch', nargs='+', default=None, type=int)
    parser.add_argument('--down-sample-train', default=None, type=int)
    parser.add_argument('--down-sample-validation', default=2000, type=int)
    parser.add_argument('--random-seed', default=42, type=int)
    parser.add_argument('--use-auth-token', action='store_true')
    parser.add_argument('--n-trials', default=10, type=int)
    parser.add_argument('--eval-step', default=100, type=int)
    parser.add_argument('--parallel-cpu', action='store_true')
    parser.add_argument('--cache-dir', default=None, type=str)
    parser.add_argument('--output-dir', default=None, type=str)
    parser.add_argument('--ray-result-dir', default=None, type=str)
    parser.add_argument('--model-alias', default=None, type=str)
    parser.add_argument('--model-organization', default=None, type=str)
    opt = parser.parse_args()

    train(model_name=opt.model_name,
          model_low_cpu_mem_usage=opt.low_cpu_mem_usage,
          dataset=opt.dataset,
          dataset_name=opt.dataset_name,
          dataset_column_question=opt.dataset_column_question,
          dataset_column_passage=opt.dataset_column_passage,
          dataset_column_answer=opt.dataset_column_answer,
          dataset_split_train=opt.dataset_split_train,
          dataset_split_validation=opt.dataset_split_validation,
          dataset_split_test=opt.dataset_split_test,
          search_range_lr=opt.search_range_lr,
          search_range_epoch=opt.search_range_epoch,
          search_list_batch=opt.search_list_batch,
          down_sample_train=opt.down_sample_train,
          down_sample_validation=opt.down_sample_validation,
          random_seed=opt.random_seed,
          use_auth_token=opt.use_auth_token,
          n_trials=opt.n_trials,
          eval_step=opt.eval_step,
          parallel_cpu=opt.parallel_cpu,
          cache_dir=opt.cache_dir,
          output_dir=opt.output_dir,
          ray_result_dir=opt.ray_result_dir,
          model_alias=opt.model_alias,
          model_organization=opt.model_organization)
