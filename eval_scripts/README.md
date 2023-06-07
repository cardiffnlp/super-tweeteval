# Official Evaluation Script

## Example: Flan-T5 baselines
```shell
python tweet_qa.py -p flan_t5_prediction_files/flan-t5-small-tweet-qa.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-qa.json
python tweet_qg.py -p flan_t5_prediction_files/flan-t5-small-tweet-qg.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-qg.json
python tempo_wic.py -p flan_t5_prediction_files/flan-t5-small-tempo-wic.txt -o flan_t5_evaluation_outputs/flan-t5-small-tempo-wic.json
python tweet_nerd.py -p flan_t5_prediction_files/flan-t5-small-tweet-nerd.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-nerd.json
python tweet_intimacy.py -p flan_t5_prediction_files/flan-t5-small-tweet-intimacy.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-intimacy.json
python tweet_similarity.py -p flan_t5_prediction_files/flan-t5-small-tweet-similarity.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-similarity.json 
python tweet_ner7.py --t2t-format -p flan_t5_prediction_files/flan-t5-small-tweet-ner7.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-ner7.json
python tweet_topic.py -p flan_t5_prediction_files/flan-t5-small-tweet-topic.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-topic.json
python tweet_sentiment.py -p flan_t5_prediction_files/flan-t5-small-tweet-sentiment.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-sentiment.json


python tweet_qa.py -p flan_t5_prediction_files/flan-t5-base-tweet-qa.txt -o flan_t5_evaluation_outputs/flan-t5-base-tweet-qa.json
python tweet_qg.py -p flan_t5_prediction_files/flan-t5-base-tweet-qg.txt -o flan_t5_evaluation_outputs/flan-t5-base-tweet-qg.json
python tempo_wic.py -p flan_t5_prediction_files/flan-t5-base-tempo-wic.txt -o flan_t5_evaluation_outputs/flan-t5-base-tempo-wic.json
python tweet_nerd.py -p flan_t5_prediction_files/flan-t5-base-tweet-nerd.txt -o flan_t5_evaluation_outputs/flan-t5-base-tweet-nerd.json
python tweet_intimacy.py -p flan_t5_prediction_files/flan-t5-base-tweet-intimacy.txt -o flan_t5_evaluation_outputs/flan-t5-base-tweet-intimacy.json
python tweet_similarity.py -p flan_t5_prediction_files/flan-t5-base-tweet-similarity.txt -o flan_t5_evaluation_outputs/flan-t5-base-tweet-similarity.json
python tweet_ner7.py --t2t-format -p flan_t5_prediction_files/flan-t5-base-tweet-ner7.txt -o flan_t5_evaluation_outputs/flan-t5-base-tweet-ner7.json
python tweet_topic.py -p flan_t5_prediction_files/flan-t5-base-tweet-topic.txt -o flan_t5_evaluation_outputs/flan-t5-base-tweet-topic.json
python tweet_sentiment.py -p flan_t5_prediction_files/flan-t5-base-tweet-sentiment.txt -o flan_t5_evaluation_outputs/flan-t5-base-tweet-sentiment.json
```