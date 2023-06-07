import logging
import argparse
import json
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# argument
parser = argparse.ArgumentParser(description='Super TweetEval evaluation script.')
parser.add_argument('-p', '--prediction-file', required=True, type=str,
                    help="a text file that contains the model prediction on the test set in each line")
parser.add_argument('-o', '--output-file', default="super_tweeteval_result.json", type=str, help="path to the output file")
opt = parser.parse_args()

# load dataset
data = load_dataset("cardiffnlp/super_tweeteval", "tweet_sentiment", use_auth_token=True, split="test")
label_names = data.features['gold_label'].names

with open(opt.prediction_file) as f:
    output = [i for i in f.read().split("\n")]
    predictions = [label_names.index(x) for x in output if x in label_names]
    
# metric: macro_averaged_mean_absolute_error
# from: https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/metrics/_classification.py 
gold_labels = np.array(data["gold_label"])

labels = [0,1,2,3,4]
mae = []
sample_weight = np.ones(gold_labels.shape)
for possible_class in labels:
    indices = np.flatnonzero(gold_labels == possible_class)
    mae.append(
        mean_absolute_error(
            gold_labels[indices],
            predictions[indices],
            sample_weight=sample_weight[indices],
        )
    )

eval_metric = {"macro_averaged_mean_absolute_error": np.sum(mae) / len(mae)}
logging.info(json.dumps(eval_metric, indent=4))
with open(opt.output_file, 'w') as f:
    json.dump(eval_metric, f)
