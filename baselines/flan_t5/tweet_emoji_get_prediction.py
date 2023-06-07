from typing import List

import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer

data = load_dataset("cardiffnlp/super_tweeteval", "tweet_emoji", split="test")
label_names = data.features['gold_label'].names
label_names = [x.split(',')[1] for x in label_names]


class Classifier:

    def __init__(self, model_name: str = None):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.id2label = {self.tokenizer.convert_tokens_to_ids(f'<emoji_{i}>')[0]: label_names[i] for i in range(100)}
        print(self.id2label)

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: List, batch_size: int = 128):
        batch_size = len(text) if batch_size is None else batch_size
        _index = list(range(0, len(text), batch_size)) + [len(text) + 1]
        pred = []
        with torch.no_grad():
            for i in range(len(_index) - 1):
                encoded_input = self.tokenizer.batch_encode_plus(text[_index[i]: _index[i+1]], return_tensors='pt', padding=True, truncation=True)
                output = self.model.generate(
                    **{k: v.to(self.device) for k, v in encoded_input.items()},
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1)
                pred_vocab = output.scores[0].argsort().tolist()
                pred_vocab = [[self.id2label[_k] for _k in k if _k in self.id2label] for k in pred_vocab]
                pred += pred_vocab
        return pred


if __name__ == '__main__':

    model = Classifier("cardiffnlp/flan-t5-small-tweet-emoji")
    prediction = model.predict(data['text'])
    with open("flan-t5-small-tweet-emoji.txt", "w") as f:
        f.write("\n".join([",".join(p) for p in prediction]))
    #
    # model = Classifier("cardiffnlp/flan-t5-base-tweet-emoji")
    # prediction = model.predict(data['text'])
    # with open("flan-t5-base-tweet-emoji.txt", "w") as f:
    #     f.write("\n".join([",".join(p) for p in prediction]))
