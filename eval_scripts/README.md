# Official Evaluation Script

## Example
```shell
python tweet_qa.py -p flan_t5_prediction/flan-t5-small-tweet-qa.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-qa.json
python tweet_qg.py -p flan_t5_prediction/flan-t5-small-tweet-qg.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-qg.json
python tempo_wic.py -p flan_t5_prediction/flan-t5-small-tempo-wic.txt -o flan_t5_evaluation_outputs/flan-t5-small-tempo-wic.json
python tweet_nerd.py -p flan_t5_prediction/flan-t5-small-tweet-nerd.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-nerd.json
python tweet_intimacy.py -p flan_t5_prediction/flan-t5-small-tweet-intimacy.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-intimacy.json
python tweet_similarity.py -p flan_t5_prediction/flan-t5-small-tweet-similarity.txt -o flan_t5_evaluation_outputs/flan-t5-small-tweet-similarity.json 
```