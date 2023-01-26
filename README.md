# indotwitterlocation

Machine learning , deep learning, and pre-trained language models approachesin estimating the location of Twitter users at the regional level.

![alt text](https://github.com/ir-nlp-csui/indotwitterlocation/blob/main/BDCC-06-00077-g003.png?raw=true)

### Prerequisites
1. Python 3.7 or higher 
2. GPU for running the models (we used NVIDIA DGX-1)

### Directory structure

Due to Twitter's data privacy policy, we are unable to display all of the data we have collected. However, we do provide a tweet id for each dataset so readers are able obtain the data using the Twitter API.

```
indotwitterlocation
|  |___ Using Twitter API to get Tweets.ipynb ................................................................... 1
|  |___ collected_tweets.csv .................................................................................... 2
|  |___ Classification.ipynb .................................................................................... 3
|  |___ LSTM-vote.ipynb ......................................................................................... 4
|  |___ LSTM-agg.ipynb .......................................................................................... 5
|  |___ LSTM-vote-word2vec.ipynb ................................................................................ 6
|  |___ IndoBERTtweet-2e5-4 epochs-Vote.ipynb ................................................................... 7
|  |___ IndoBERTtweet-2e5-4 epochs-Agg.ipynb .................................................................... 8
|  |___ indoLEM2e-5batch16-Vote.ipynb ........................................................................... 9
|  |___ indoLEM2e-5batch16-Agg.ipynb ............................................................................ 10
|  |___ indoNLU2e-5batch16-Vote.ipynb ........................................................................... 11
|  |___ indoNLU2e-5batch16-person.ipynb ......................................................................... 12
|  |___ indoNLU2e-5batch16-nonperson.ipynb ...................................................................... 13
|  |___ indoNLU2e-5batch16-Agg.ipynb ............................................................................ 14
|  |___ periksaa.csv ............................................................................................ 15
|___ dataset/
|      |___ new/
|          |___ finaluser.csv ................................................................................... 16
|      |___ person_nonperson/ ................................................................................... 17
|          |___ test_df.csv
|          |___ userlvlval.csv
|          |___ val_df.csv
|          |___ train_df.csv
|          |___ userlvltest.csv
|          |___ userlvltrain.csv
| 
|___ newdata/ ................................................................................................... 18
|     |___ dev.csv  
|     |___ train.csv 
|     |___ train.csv 
|___ lstm_result/
|     |___ lstm_majority_vote_word2vec.csv ...................................................................... 19
|     |___ lstm_majority_vote.csv ............................................................................... 20
|
|___ bert_result/ ............................................................................................... 21
|     |___ indoNLU2e_5ResultsVote.csv.csv  
|     |___ indoLEM_model_2e_5_batch16.csv 
|     |___ indoNLU2e_5Results.csv 
|     |___ indoBERTweet2e_5ResultsVote.csv 
|     |___ indoLEM2e_5ResultsVote.csv 
|     |___ indoNLU2e_5ResultsVote.csv 
|     |___ indoBERTweet2e_5Results.csv  
|     |___ indoBERTweetTweetslevel.csv  
|
|
|___ nonpersondata/ ............................................................................................. 22
|     |___ dev.csv 
|     |___ train.csv 
|     |___ test.csv 
|
|___ persondata/ ................................................................................................ 23
|     |___ dev.csv 
|     |___ train.csv 
|     |___ test.csv 
|
|___ newdataVote/ ............................................................................................... 24
|     |___ dev.csv 
|     |___ train.csv  
|     |___ test.csv 
|
|___ NER/ ....................................................................................................... 25
|     |___ LSTM-vote.ipynb ...................................................................................... 25
|     |___ IndoBERTtweet-2e5-4 epochs-Agg.ipynb ................................................................. 25
|     |___ Classification.ipynb ................................................................................. 25
|     |___ LSTM-agg.ipynb ....................................................................................... 25
|     |___ indoNLU2e-5batch16-Vote.ipynb ........................................................................ 25
|     |___ indoLEM2e-5batch16-Agg.ipynb ......................................................................... 25
|     |___ indoNLU2e-5batch16-Agg.ipynb ......................................................................... 25
|     |___ indoLEM2e-5batch16-Vote.ipynb ........................................................................ 25
|     |___ IndoBERTtweet-2e5-4 epochs-Vote.ipynb ................................................................ 25
|     |___ LSTM-vote-word2vec.ipynb ............................................................................. 25
|      |___ original_data/ ...................................................................................... 25
|          |___ test_df.csv ..................................................................................... 25
|          |___ userlvlval.csv .................................................................................. 25
|          |___ val_df.csv ...................................................................................... 25
|          |___ train_df.csv .................................................................................... 25
|          |___ userlvltest.csv ................................................................................. 25
|          |___ userlvltrain.csv ................................................................................ 25
|      |___ DataWithNERConcat/ .................................................................................. 25
|          |___ test_df.csv ..................................................................................... 25
|          |___ val_df.csv ...................................................................................... 25
|          |___ train_df.csv .................................................................................... 25
|          |___ tweetAgg/ ....................................................................................... 25
|              |___ aggval_df.csv ............................................................................... 25
|              |___ aggtrain_df.csv ............................................................................. 25
|              |___ aggtest_df.csv .............................................................................. 25
|      |___ create_dataset/ ..................................................................................... 25
|          |___ createDataset.ipynb ............................................................................. 25
|          |___ data/ ........................................................................................... 25
|              |___ test_informal.txt ........................................................................... 25
|              |___ train.txt ................................................................................... 25
|              |___ readme.txt .................................................................................. 25
|              |___ dev.txt ..................................................................................... 25
|              |___ test_formal.txt ............................................................................. 25
|      |___ bert_result/ ........................................................................................ 25
|          |___ dev.csv ......................................................................................... 25
|          |___ train.csv ....................................................................................... 25
|          |___ test.csv ........................................................................................ 25
|      |___ DataWithNER/ ........................................................................................ 25
|          |___ train_text.csv .................................................................................. 25
|          |___ val_text.csv .................................................................................... 25
|          |___ test_text.csv ................................................................................... 25
|
|___ revision_dataset/ .......................................................................................... 26
|     |___ test_df.csv .......................................................................................... 27
|     |___ userlvlval.csv ....................................................................................... 28
|     |___ val_df.csv ........................................................................................... 29
|     |___ train_df.csv ......................................................................................... 30
|     |___ userlvltest.csv ...................................................................................... 31
|     |___ userlvltrain.csv ..................................................................................... 32
|      |___ tweetAgg/ ........................................................................................... 33
|          |___ aggval_df.csv ................................................................................... 34
|          |___ aggtrain_df.csv ................................................................................. 35
|          |___ aggtest_df.csv .................................................................................. 36

```
1. Code for getting user tweeets by annotated user.   
2. Collected tweets from points 1
3. Experiments with machine learnings and link to almost all experiments
4. Majority vote scenario using LSTM
5. Text aggregation as one scenario using LSTM
6. Majority vote scenario using LSTM with word2vec embedding
7. Majority vote scenario using IndoBERTweet
8. Text aggregation as one scenario using IndoBERTweet
9. Majority vote scenario using IndoBERT (IndoLEM)
10. Text aggregation as one scenario using IndoBERT (IndoLEM)
11. Majority vote scenario using IndoBERT (IndoNLU)
12. Text aggregation as one scenario using IndoBERT (IndoNLU)
13. Majority vote scenario using IndoBERT for person user (IndoNLU)
14. Majority vote scenario using IndoBERT for nonperson user (IndoNLU)
15. Text aggregation as one scenario using IndoBERT for person as well as non person user (IndoNLU)
16. Data set consisting of users and their locations
17. Data set consisting of users, usernames, descriptions, and their locations for person and non person 
18. Data set for aggregation as one scenario using BERT 
19. Store the results of experiments using LSTM
20. Store the results of experiments using LSTM
21. Store the results of experiments using BERT 
22. Data set only for non person data (for crosstesting experiments) 
23. Data set only for non person data (for crosstesting experiments)
24. Data set for majority vote scenario using BERT 
25. Experiments with NER 
26-36. Revision dataset (updated version of dataset number 17)

## Built With

* [scikit-learn](https://scikit-learn.org/stable/)
* [Transformers](https://huggingface.co/docs/transformers/) 
* [Pytorch](https://pytorch.org/)

## Research Paper

```
@Article{bdcc6030077,
AUTHOR = {Simanjuntak, Lihardo Faisal and Mahendra, Rahmad and Yulianti, Evi},
TITLE = {We Know You Are Living in Bali: Location Prediction of Twitter Users Using BERT Language Model},
JOURNAL = {Big Data and Cognitive Computing},
VOLUME = {6},
YEAR = {2022},
NUMBER = {3},
ARTICLE-NUMBER = {77},
URL = {https://www.mdpi.com/2504-2289/6/3/77},
ISSN = {2504-2289},
ABSTRACT = {Twitter user location data provide essential information that can be used for various purposes. However, user location is not easy to identify because many profiles omit this information, or users enter data that do not correspond to their actual locations. Several related works attempted to predict location on English-language tweets. In this study, we attempted to predict the location of Indonesian tweets. We utilized machine learning approaches, i.e., long-short term memory (LSTM) and bidirectional encoder representations from transformers (BERT) to infer Twitter users&rsquo; home locations using display name in profile, user description, and user tweets. By concatenating display name, description, and aggregated tweet, the model achieved the best accuracy of 0.77. The performance of the IndoBERT model outperformed several baseline models.},
DOI = {10.3390/bdcc6030077}
}

```
