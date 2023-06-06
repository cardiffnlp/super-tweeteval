import logging
import argparse
import json
from statistics import mean
from datasets import load_dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# argument
parser = argparse.ArgumentParser(description='Super TweetEval evaluation script.')
parser.add_argument('-p', '--prediction-file', required=True, type=str,
                    help="a text file that contains the model prediction on the test set in each line")
parser.add_argument('-o', '--output-file', default="super_tweeteval_result.json", type=str, help="path to the output file")
opt = parser.parse_args()

# load dataset
data = load_dataset("cardiffnlp/super_tweeteval", "tempo_wic", use_auth_token=True, split="test")

# metric
label2id = {"no": 0, "yes": 1}
with open(opt.prediction_file) as f:
    _predictions = [label2id[i] if i in label2id else i for i in f.read().split("\n")]
    _references = data["gold_label_binary"]
eval_metric = {"accuracy": mean(int(a == b) for a, b in zip(_predictions, _references))}
logging.info(json.dumps(eval_metric, indent=4))
with open(opt.output_file, 'w') as f:
    json.dump(eval_metric, f)
