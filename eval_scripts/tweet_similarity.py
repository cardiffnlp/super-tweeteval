import logging
import argparse
import json
from datasets import load_dataset
from evaluate import load

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# argument
parser = argparse.ArgumentParser(description='Super TweetEval evaluation script.')
parser.add_argument('-p', '--prediction-file', required=True, type=str,
                    help="a text file that contains the model prediction on the test set in each line")
parser.add_argument('-o', '--output-file', default="super_tweeteval_result.json", type=str, help="path to the output file")
opt = parser.parse_args()

# load dataset
data = load_dataset("cardiffnlp/super_tweeteval", "tweet_similarity", use_auth_token=True, split="test")

# metric
metric = load("spearmanr")
with open(opt.prediction_file) as f:
    _predictions = []
    for i in f.read().split("\n"):
        try:
            _predictions.append(float(i))
        except ValueError:
            _predictions.append(0)
    _references = data["gold_score"]
eval_metric = metric.compute(predictions=_predictions, references=_references)
logging.info(json.dumps(eval_metric, indent=4))
with open(opt.output_file, 'w') as f:
    json.dump(eval_metric, f)
