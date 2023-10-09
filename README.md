# SmsClassificationByBert

### Some description about the code below

- I use the model in huggingface `twitter-roberta-base` and then dine-tuned it.
- However, I can not run the train code in ipynb, so I create a new file to run it.
- Here are the details of my env.
    - python  --3.8
    - torch --1.10.0
    - cuda --11.3
    - transformers --4.6.1

- Notice: To run the code below, you need to load the model from https://huggingface.co/cardiffnlp/twitter-roberta-base


### About the task

- It's a task of SMS message classification. There are 3 lables about the text [0, 1, 2] which represent nomal, spam and smishing respectively
- So that we need to train a classifier to predict the class from the text.
