import logging
import argparse
import json
from datasets import load_dataset
from sklearn.metrics import f1_score

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# argument
parser = argparse.ArgumentParser(description='Super TweetEval evaluation script.')
parser.add_argument('-p', '--prediction-file', required=True, type=str,
                    help="a text file that contains the model prediction on the test set in each line")
parser.add_argument('-o', '--output-file', default="super_tweeteval_result.json", type=str, help="path to the output file")
opt = parser.parse_args()

# load dataset
data = load_dataset("cardiffnlp/super_tweeteval", "tweet_topic", use_auth_token=True, split="test")
label_names = data.features['gold_label_list'].feature.names

with open(opt.prediction_file) as f:
    lines = f.read().split('\n')
    predictions = []
    for l in lines:
        pred_instance = [0] * len(label_names)
        for label in l.split(','):
            if label in label_names:
                pred_instance[label_names.index(label)] = 1

        predictions.append(pred_instance)
    

# metric
gold_labels = data["gold_label_list"]
eval_metric = {"f1-macro": f1_score(gold_labels, predictions, average='macro')}
logging.info(json.dumps(eval_metric, indent=4))
with open(opt.output_file, 'w') as f:
    json.dump(eval_metric, f)
