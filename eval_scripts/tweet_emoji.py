import logging
import argparse
import json
from datasets import load_dataset
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# argument
parser = argparse.ArgumentParser(description='Super TweetEval evaluation script.')
parser.add_argument('-p', '--prediction-file', required=True, type=str,
                    help="a text file that contains the model prediction on the test set in each line")
parser.add_argument('-o', '--output-file', default="super_tweeteval_result.json", type=str, help="path to the output file")
opt = parser.parse_args()

# load dataset
data = load_dataset("cardiffnlp/super_tweeteval", "tweet_emoji", use_auth_token=True, split="test")
label_names = data.features['gold_label'].names
label_names = [x.split(',')[1] for x in label_names]

with open(opt.prediction_file) as f:
    lines = f.read().split('\n')
    predictions = []
    for l in lines:
        pred_instance = []
        # consider only top 5 predictions
        for label in l.split(',')[:5]: 
            if label in label_names:
                pred_instance.append(label_names.index(label))
            else:
                pred_instance.append(-1)  # emoji not in label_names

        predictions.append(pred_instance)
    
# metric: accuracy at top 5
gold_labels = np.array(data["gold_label"])

eval_metric = {"accuracy_top5": np.mean([1 if gold_labels[i] in predictions[i] else 0 for i in range(len(gold_labels))])}
logging.info(json.dumps(eval_metric, indent=4))
with open(opt.output_file, 'w') as f:
    json.dump(eval_metric, f)
