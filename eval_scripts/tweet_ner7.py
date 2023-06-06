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
parser.add_argument('--t2t-format', action="store_true", help="path to the output file")
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
if opt.t2t_format:
    # format prediction file in IOB sequence
    with open(opt.prediction_file) as f:
        output = [list(set(i.split(","))) for i in f.read().split("\n")]
    prediction_sequence = []
    for d, o in zip(data, output):
        tag_seq = ['O'] * len(d['text_tokenized'])
        for _o in o.split(","):
            if len(_o.split(":")) != 2:
                continue
            entity, _type = _o.split(":")
            entity_tokens = entity.split(" ")
            try:
                i = d['text_tokenized'].index(entity_tokens[0])
                tag_seq[i] = f"B-{_type}"
                if len(entity_tokens) > 1:
                    for j in range(1, len(entity_tokens)):
                        tag_seq[i + j] = f"I-{_type}"
            except ValueError:
                continue
        prediction_sequence.append(tag_seq)
else:
    with open(opt.prediction_file) as f:
        prediction_sequence = [[id2label[j] if j in id2label else j for j in i.split('\t')] for i in f.read().split("\n")]

eval_metric = metric.compute(predictions=prediction_sequence, references=true_sequence)
eval_metric = {'overall_f1': eval_metric['overall_f1']}
logging.info(json.dumps(eval_metric, indent=4))
with open(opt.output_file, 'w') as f:
    json.dump(eval_metric, f)
