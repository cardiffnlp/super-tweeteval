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
data = load_dataset("cardiffnlp/super_tweeteval", "tweet_ner7", use_auth_token=True, split="test")
labels = [
    'B-corporation', 'B-creative_work', 'B-event', 'B-group', 'B-location', 'B-person', 'B-product',
    'I-corporation', 'I-creative_work', 'I-event', 'I-group', 'I-location', 'I-person', 'I-product', 'O']
id2label = {i: label for i, label in enumerate(labels)}
true_sequence = [[id2label[i] for i in ii] for ii in data['gold_label_sequence']]

# metric
metric = load("seqeval")
with open(opt.prediction_file) as f:
    _predictions = [[id2label[j] if j in id2label else j for j in i.split('\t')] for i in f.read().split("\n")]
eval_metric = metric.compute(predictions=_predictions, references=true_sequence)
eval_metric = {'overall_f1': eval_metric['overall_f1']}
logging.info(json.dumps(eval_metric, indent=4))
with open(opt.output_file, 'w') as f:
    json.dump(eval_metric, f)
