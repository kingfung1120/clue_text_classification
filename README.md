# Chinese Text Classification using BERT

Implemented with Pytorch 

## Parameters

- Padding size: 512
- Bacth size: 8
- Learning rate: 1e-5
- Epoch: 2

training takes around 17mins/epoch using GPU from google colab

## Envrionment

- Python 3.7
- Please refer to requirements.txt

## Dataset

Chinese GLUE public challenge

- dataset is stored in iflytek_public
- train: train.json (70% is being used as training data, the remaining is used as validation data)
- test: test.json
- class label: labels.json (119 classes)

## Result 

Model|Accuracy|Remarks 
--|--|--
bert|0.57|
bert_CNN|0.58|bert with convolution layer

## Instruction

```

# change directory into src/

cd src

# train bert

python run.py --model=bert

# train bert_cnn

python run.py --model=bert_cnn

# training performance will be stored as result.json in exp_dump
# model will be evaluated on validation set every 100 training iterations

```

## Reference

[1] Attention Is All You Need

[2] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
