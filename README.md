# smart_crypto_crawler
This simple project consists of an end-to-end AI pipeline that performs technical and fundamental analysis of different cryptocurrency

## 🟢 Phase 1:
#### We have designed two end-to-end crawlers using selenium  for fetching the latest news from Coinmarketcap.com and Cointelegraph.com websites 


## 🟢 Phase 2:design 
#### We did a comprehensive method for technical analysis of different cryptocurrencies prices and volumes also we wrote different deep model for doing text processing task using LLMs for fundemnetal analysis 
### <b> ✅ fundamental analysis : </b>
#### We have different modules for each of our tasks :
#### 1- News sentiment analysis ( we perform sentiment analysis using the ensemble method with Fine-Bert and Deberta for each retrieved article, and also perform a weekly and monthly basis analysis too! )
#### 2- News text summarization ( for this pourpose we have used Bart LLM which is one of the best models we have tested)
#### 3- keyword extraction ( we have used KeyBert LLM for doing this)
#### 4-Translation of summary (after searching for an efficient LLM for long input handling, we have used the "mbart-large-50-many-to-many-mmt" model for English to Persian translation)
### <b>✅ technical analysis : </b>
#### We have different modules for each of our tasks :
#### 1- we have tried different deep models for time series prediction, finally, we have used Temporal-CNN and Auto-regressive RNN in an ensemble method for price and volume prediction (based on the last 60 days ago) 

## 🟢 Phase 3:
#### We have designed a script for scheduling the crawler and main code of the project and sending results to the telegram bot and telegram channel too!

<a href="https://uupload.ir/view/smart_crypto_crawler_demo_gtu2.png" target="_blank">
    <img src="https://s8.uupload.ir/files/smart_crypto_crawler_demo_gtu2.png" border="0" width="350" height="600">
</a>


### 👉 this is just a demo,asking to design a more complex and sophisticated version of this project,  please message me: at mohsen.hamiii@gmail.com


