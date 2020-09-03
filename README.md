# Try to Implement BERT for Text Classification
My eighth project on github. Try to use transformers and BERT to classify tweets into different categories.

Feel free to provide comments, I just started learning Python for 3 monnths and I am now concentrating on data anylysis and web presentation.

## Methodology
transformers provided a very simple implementation for BERT, and thanks to Tohoku University (東北大学) which has a SOTA pre-trained model. This project will use cl-tohoku/bert-base-japanese, for the dataset I used a twitter dataset in Japanese provided by Suzuki Laboratory (link provided below), it is a dataset containing tweets content and labels of positive or negative review labeled manually, which I think is a very good dataset for a sanity test of text classification project. Because of copyright they only provide the tweet ID so an API to extract those tweets is needed, thus the step would be like this:
1. Extract tweets from Twitter using API (twitter_extract.py) (Application for Twitter API is necessary)
2. Store tweets and labels to database (twitter_load.py) (SQLite used here)
3. Extract tweet contents from database by different conditions and transform to training texts and labels (twitter_bert_2class.py)
4. Use cl-tohoku/bert-base-japanese to train and test (twitter_bert_2class.py)

The database is like this. it contains 9 categories representing 9 different products, and 5 types of reviews (e.g. positive, negative, neutral, etc.) but for simplicity here I will just focus on positive and negative review
![image](https://github.com/leolui2004/bert_classify_try/blob/master/twitter.png)

## Result
Because the result is insanely impressive so I would like to show the result picture first
![image](https://github.com/leolui2004/bert_classify_try/blob/master/bert_test_v1.png)

I did two tests overall, the first test only using one topic category with only positive and negative reviews, respective SQL script like this:
```
SELECT text,label FROM tw WHERE topic='10020' AND (label='01000' OR label='00100') ORDER BY RANDOM()
```
I chose 30% of the data for testing, thus the sample train size is 359 and the sample test size is 153.
The model was trained with batch size 64 and 4 epochs, below is the result.
```
Epoch 1/4
8/8 [==============================] - 4s 479ms/step - loss: 0.6806 - acc: 0.6445
Epoch 2/4
8/8 [==============================] - 4s 480ms/step - loss: 0.5063 - acc: 0.7656
Epoch 3/4
8/8 [==============================] - 4s 481ms/step - loss: 0.3949 - acc: 0.8242
Epoch 4/4
8/8 [==============================] - 4s 481ms/step - loss: 0.2633 - acc: 0.9004
Accuracy: 0.93464
```

For the second test, I tried to use two topic categories, however as the size is too large that even I trained on Colab Pro its still occur OOM error, thus I limited the number of sample size to 500 and also maximum sentence length to 100. SQL script like this:
```
SELECT text,label FROM tw WHERE (topic='10001' OR topic='10020') AND (label='01000' OR label='00100') ORDER BY RANDOM() LIMIT 500
```
The sample train size would be 350 and the sample test size would be 150.
Again the model was trained with batch size 64 and 4 epochs, below is the result.
```
Epoch 1/4
8/8 [==============================] - 3s 347ms/step - loss: 0.7332 - acc: 0.5300
Epoch 2/4
8/8 [==============================] - 3s 345ms/step - loss: 0.6262 - acc: 0.6500
Epoch 3/4
8/8 [==============================] - 3s 346ms/step - loss: 0.5023 - acc: 0.7600
Epoch 4/4
8/8 [==============================] - 3s 346ms/step - loss: 0.3370 - acc: 0.8920
Accuracy: 0.97333
```

So it seems that simply implementing a pre-trained model with a little bit fine-tuning (the training process takes less than a minute on Colab Pro) would provide a fabulous result that cannot be simply archived by models in the past. However this time is just a little try to see if anything works, next time I will try to do further implementation and testing.

## Reference
Twitter日本語評判分析データセット
http://www.db.info.gifu-u.ac.jp/data/Data_5d832973308d57446583ed9f
