# Implementing BERT for text classification and visualization
My eighth project on github. Using transformers and BERT to classify tweets into different categories and visualizing sentences on embedding projector.

Feel free to provide comments, I just started learning Python for 3 monnths and I am now concentrating on data anylysis and web presentation.

## Classification

### Methodology
transformers provided a very simple implementation for BERT, and thanks to Tohoku University (東北大学) which has a SOTA pre-trained model. This project will use cl-tohoku/bert-base-japanese, for the dataset I used a twitter dataset in Japanese provided by Suzuki Laboratory (link provided below), it is a dataset containing tweets content and labels of positive or negative review labeled manually, which I think is a very good dataset for a sanity test of text classification project. Because of copyright they only provide the tweet ID so an API to extract those tweets is needed, thus the step would be like this:
1. Extract tweets from Twitter using API (twitter_extract.py) (Application for Twitter API is necessary)
2. Store tweets and labels to database (twitter_load.py) (SQLite used here)
3. Extract tweet contents from database by different conditions and transform to training texts and labels (twitter_bert_2class.py)
4. Use cl-tohoku/bert-base-japanese to train and test (twitter_bert_2class.py)

The database is like this. it contains 9 categories representing 9 different products, and 5 types of reviews (e.g. positive, negative, neutral, etc.) but for simplicity here I will just focus on positive and negative review
![image](https://github.com/leolui2004/bert_classify_try/blob/master/twitter.png)

### Result
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

### Reference
Twitter日本語評判分析データセット
http://www.db.info.gifu-u.ac.jp/data/Data_5d832973308d57446583ed9f

### Addition Part
Today just found another dataset which contains livedoor news content with different categories. I selected 6 of them, just pick the topic of each news and label by category. And the number of samples for each category is quite evenly distributed (e.g. 871, 865, 871, 843, 829, 901).

This time two different tests were done, one is just same as before, the second one I added 3 additional layers to the fine-tuning part, a 0.3 Dropout, a 64-unit Dense with relu, and a 0.3 Dropout. It is sad that because just after 3 epoch the test accuracy is already over 90%, adding more layers cannot see any improvement, but at least I tested on it and it seems work without any problem.

Respective file: livedoor_preprocess.py, livedoor_bert.py

No additional fine-tuning layer
```
Epoch 1/3
162/162 [==============================] - 21s 131ms/step - loss: 0.8841 - acc: 0.6622
Epoch 2/3
162/162 [==============================] - 21s 131ms/step - loss: 0.3737 - acc: 0.8685
Epoch 3/3
162/162 [==============================] - 21s 131ms/step - loss: 0.1611 - acc: 0.9459
Accuracy: 0.9936
```

With additional fine-tuning layer
```
Epoch 1/3
162/162 [==============================] - 21s 131ms/step - loss: 1.1912 - acc: 0.5239
Epoch 2/3
162/162 [==============================] - 21s 131ms/step - loss: 0.6025 - acc: 0.7772
Epoch 3/3
162/162 [==============================] - 21s 132ms/step - loss: 0.3497 - acc: 0.8894
Accuracy: 0.9562
```

For reference, the dataset can be found here:
https://www.rondhuit.com/download.html

## Visualization

### Methodology
Again using the Twitter database, this time another pre-trained model is used, the model is trained by BERT with SentencePiece using Wikipedia Japanese. The articles first divided into sentences, and then each sentence is vectorized using BERT with SentencePiece, the result was saved to TSV which is a format required by Tensorflow Embedding Projector. Text Length is limited to 10-200 and the total sentences processed are 868.

Respective file: twitter_visual.py

### Result
The whole picture is like this, PCA is used
![image](https://github.com/leolui2004/bert_classify_visualize/blob/master/twitter_visual_1.png)

Example 1 - Group of sentences mentioning the movement of the cleaning robot
![image](https://github.com/leolui2004/bert_classify_visualize/blob/master/twitter_visual_2.png)

Example 2 - Group of sentences mentioning the price of the cleaning robot
![image](https://github.com/leolui2004/bert_classify_visualize/blob/master/twitter_visual_3.png)

The result is not too good, but at least here can see some of the relationship between sentences found by the model, more pre-processing should be better for the model to calculate the vector.

### Reference
Embedding Projector
https://projector.tensorflow.org/
BERT with SentencePiece を日本語 Wikipedia で学習してモデルを公開しました
https://yoheikikuta.github.io/bert-japanese/
