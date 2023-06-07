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
data = load_dataset("cardiffnlp/super_tweeteval", "tweet_hate", use_auth_token=True, split="test")
label_names = data.features['gold_label'].names

with open(opt.prediction_file) as f:
    output = [i for i in f.read().split("\n")]
    predictions = [label_names.index(x) for x in output if x in label_names]
    
# metric: ((macro_f1 for binary (hate/not hate) classification) + (micro_f1 for multi-class classification (only hate))) /2
gold_labels = data["gold_label"]
# do not consider not_hate class
f1_multi = f1_score(gold_labels, predictions, labels=list(range(7)), average='micro')

# consider all hate subclasses as one class
predictions_binary = [1 if x in list(range(7)) else 0 for x in predictions]
gold_labels_binary = [1 if x in list(range(7)) else 0 for x in gold_labels]
f1_binary = f1_score(gold_labels_binary, predictions_binary, average='macro')


eval_metric = {"(f1_bin + f1_multi)/2": (f1_multi+f1_binary)/2}
logging.info(json.dumps(eval_metric, indent=4))
with open(opt.output_file, 'w') as f:
    json.dump(eval_metric, f)
