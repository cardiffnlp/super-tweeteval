# Fine-tune T5 on SuperTweetEval
Model training example for `google/flan-t5-large`.
```shell
git clone https://github.com/cardiffnlp/super-tweeteval
cd super-tweeteval/baselines/flan_t5

# tempo_wic
python tempo_wic.py -m google/flan-t5-large --model-alias "flan-t5-large-tempo-wic" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tempo-wic"

# tweet_emoji
python tweet_emoji.py -m google/flan-t5-large --model-alias "flan-t5-large-tweet-emoji" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-emoji"

# tweet_emotion
python tweet_emotion.py -m google/flan-t5-large--model-alias "flan-t5-large-tweet-emotion" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-emotion"

# tweet_hate
python tweet_hate.py -m google/flan-t5-large --model-alias "flan-t5-large-tweet-hate" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-hate"

# tweet_intimacy
python tweet_intimacy.py -m google/flan-t5-large --model-alias "flan-t5-large-tweet-intimacy" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-intimacy"

# tweet_ner7
python tweet_ner7.py -m google/flan-t5-large --model-alias "flan-t5-large-tweet-ner7" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-ner7"

# tweet_nerd
python tweet_nerd.py -m google/flan-t5-large --model-alias "flan-t5-large-tweet-nerd" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-nerd"

# tweet_qa
python tweet_qa.py -m google/flan-t5-large --model-alias "flan-t5-large-tweet-qa" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-qa"

# tweet_qg
python tweet_qg.py -m google/flan-t5-large --model-alias "flan-t5-large-tweet-qg" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-qg"

# tweet_sentiment
python tweet_sentiment.py -m google/flan-t5-large --model-alias "flan-t5-large-tweet-sentiment" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-sentiment"

# tweet_similarity
python tweet_similarity.py -m google/flan-t5-large --model-alias "flan-t5-large-tweet-similarity" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-similarity"

# tweet_topic
python tweet_topic.py -m google/flan-t5-large --model-alias "flan-t5-large-tweet-topic" --use-auth-token --model-organization "cardiffnlp" --search-list-batch 64 128
rm -rf ray
rm -rf ckpt
rm -rf "flan-t5-large-tweet-topic"
```

## Models
- QA
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-qa
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-qa
- QG 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-qg
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-qg
- Sentiment 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-sentiment
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-sentiment
- TempoWiC 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tempo-wic
  - https://huggingface.co/cardiffnlp/flan-t5-base-tempo-wic
- NERD 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-nerd
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-nerd
- Hate 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-hate
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-hate
- Intimacy 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-intimacy
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-intimacy
- Similarity 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-similarity
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-similarity
- Emoji 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-emoji
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-emoji
- Emotion 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-emotion
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-emotion
- Topic 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-topic
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-topic
- NER 
  - https://huggingface.co/cardiffnlp/flan-t5-small-tweet-ner7
  - https://huggingface.co/cardiffnlp/flan-t5-base-tweet-ner7

Prediction files are [here](https://github.com/cardiffnlp/super-tweeteval/tree/main/eval_scripts/flan_t5_prediction_files).