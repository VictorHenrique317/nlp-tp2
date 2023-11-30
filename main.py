#!/usr/bin/env python
# coding: utf-8

# # 2 - Fine-tuning Pretrained Transformers for PoS Tagging
# 
# ## Introduction
# 
# In the previous notebook we showed how to use a BiLSTM with pretrained GloVe embeddings for PoS tagging. In this notebook we'll be using a pretrained [Transformer](https://arxiv.org/abs/1706.03762) model, specifically the pre-trained [BERT](https://arxiv.org/abs/1810.04805) model. Our model will be composed of the Transformer and a simple linear layer.
# 
# ## Preparing Data
# 
# First, let's import the necessary Python modules.
# In google colab you will need to install the `transformers` library.
# 
# ```
# !pip install transformers
# ```

# In[1]:


# %pip install torchtext


# In[3]:


import torch
import torchtext
print(torchtext.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import data
from torchtext import datasets
from torchtext.data import Example, Dataset

from transformers import BertTokenizer, BertModel

import numpy as np

import time
import random
import functools

import os

from datasets import load_dataset


# Next, we'll set the random seeds for reproducability.

# In[253]:


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Then, we'll import the BERT tokenizer. This defines how text into the model should be processed, but more importantly contains the vocabulary that the BERT model was pretrained with. We'll be using the `bert-base-uncased` tokenizer and model. This was trained on text that has been lowercased.
# 
# In order to use pretrained models for NLP the vocabulary used needs to exactly match that of the pretrained model.

# In[254]:


tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')


# Another thing that we need to do is make sure the input sequence is formatted in the same way in which the BERT model was trained. 
# 
# BERT was trained on sequences that begin with a `[CLS]` token.
# 
# So the sequence of tokens
# 
# ```python
# text = ['jack', 'went', 'to', 'the', 'shop']
# ```
# 
# should become:
# 
# ```python
# text = ['[CLS]', 'jack', 'went', 'to', 'the', 'shop']
# ```
# 
# Along with making our vocabularies match we also need to make sure our padding and unk tokens match those used in the pretrained model. By default TorchText uses `<pad>` and `<unk>`, but the BERT model uses `[PAD]` and `[UNK]`.
# 
# Let's get the special tokens:

# In[255]:


init_token = tokenizer.cls_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

print(init_token, pad_token, unk_token)


# We are mainly interested in the actual integer representations of the special tokens. This is because we aren't using TorchText's vocabulary module, but using the one provided by the pretrained model. 
# 
# We get the indexes of the special tokens by passing them through the tokenizer's `convert_tokens_to_ids` function.

# In[256]:


init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

print(init_token_idx, pad_token_idx, unk_token_idx)


# One other thing is that the pretrained model was trained on sequences up to a maximum length and we need to ensure that our sequences are also trimmed to this length.

# In[257]:


max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

print(max_input_length)


# Next, we'll define two helper functions that make use of our vocabulary.
# 
# The first will cut the sequence of tokens to the desired maximum length, specified by our pretrained model, and then convert the tokens into indexes by passing them through the vocabulary. This is what we will use on our input sequence we want to tag.
# 
# Note that we actually cut tokens to `max_input_length-1`, this is because we need to add the special `[CLS]` token to the start of the sequence.

# In[258]:


def cut_and_convert_to_id(tokens, tokenizer, max_input_length):
    tokens = tokens[:max_input_length-1]
    tokens = [tokenizer.convert_tokens_to_ids(token) 
              if token in tokenizer.vocab 
              else tokenizer.convert_tokens_to_ids('<unk>') 
              for token in tokens]
    return tokens


# The second helper function simply cuts the sequence to the maximum length. This is used for our tags. We do not pass the tags through pretrained model's vocabulary as the vocab was only built for English sentences, and not for part-of-speech tags. We will be building the tag vocabulary ourselves.

# In[259]:


def cut_to_max_length(tokens, max_input_length):
    tokens = tokens[:max_input_length-1]
    return tokens


# We need to pass the above two functions to the `Field`, the TorchText abstraction that handles a lot of the data processing for us. We make use of Python's `functools` that allow us to pass functions which already have some of their arguments supplied. 

# In[260]:


text_preprocessor = functools.partial(cut_and_convert_to_id,
                                      tokenizer = tokenizer,
                                      max_input_length = max_input_length)

tag_preprocessor = functools.partial(cut_to_max_length,
                                     max_input_length = max_input_length)


# Next, we define our fields.
# 
# For the `TEXT` field, which will be processing the sequences we want to tag, we first tell TorchText that we do not want to use a vocabulary with `use_vocab = False`. As our model is `uncased`, we also want to ensure all text is lowercased with `lower=True`. The `preprocessing` argument is a function applied to sequences after they have been tokenized, but before they are numericalized. As we have set `use_vocab` to false, they will never actually be numericalized, and as we are using TorchText's POS datasets they have also already been tokenized - so the argument to this will just be applied to the sequence of tokens. This is where our help functions from above come in handy and `text_preprocessor` will both numericalize our data using the pretrained model's vocabulary, as well as cutting it to the maximum length. The remaining four arguments define the special tokens required by the pretrained model.
# 
# For the `UD_TAGS` field, we need to ensure the length of our tags matches the length of our text sequence. As we have added a `[CLS]` token to the beginning of the text sequence, we need to do the same with the sequence of tags. We do this by adding a `<pad>` token to the beginning which we will later tell our model to not use when calculating losses or accuracy. We won't have unknown tags in our sequence of tags, so we set the `unk_token` to `None`. Finally, we pass our `tag_preprocessor` defined above, which simply cuts the tags to the maximum length our pretrained model can handle.

# In[261]:


TEXT = data.Field(use_vocab = False,
                  lower = True,
                  preprocessing = text_preprocessor,
                  init_token = init_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

UD_TAGS = data.Field(unk_token = None,
                     init_token = '<pad>',
                     preprocessing = tag_preprocessor)


# Then, we define which of our fields defined above correspond to which fields in the dataset.

# In[262]:


fields = (("tokens", TEXT), ("pos_tags", UD_TAGS))


# Next, we load the data using our fields.

# In[296]:


def toPytorchDataset(dataset, train_set=None):
    dataset = [(example['tokens'], example['pos_tags']) for example in dataset]

    examples = [Example.fromlist([text, tags], fields=[('text', TEXT), ('udtags', UD_TAGS)]) for text, tags in dataset]
    dataset = Dataset(examples, fields=[('text', TEXT), ('udtags', UD_TAGS)])

    return dataset

nb_training_samples = 50
nb_validation_samples = int(nb_training_samples * 0.05)
# nb_validation_samples = 50
nb_test_samples = int(nb_training_samples * 0.25)
# nb_test_samples = 50

dataset = load_dataset('mac_morpho')
# train_data_raw = dataset['train'].select(range(nb_training_samples))
train_data_raw = dataset['train']
# valid_data_raw = dataset['validation'].select(range(nb_validation_samples))
valid_data_raw = dataset['validation']
# test_data_raw = dataset['test'].select(range(nb_test_samples))
test_data_raw = dataset['test']

train_data = toPytorchDataset(train_data_raw)
valid_data = toPytorchDataset(valid_data_raw, train_set=train_data_raw)
test_data = toPytorchDataset(test_data_raw, train_set=train_data_raw)

print(len(train_data.examples))
print(len(valid_data.examples))
print(len(test_data.examples))


# We can check an example by printing it. As we have already numericalized our `text` using the vocabulary of the pretrained model, it is already a sequence of integers. The tags have yet to be numericalized. 

# In[297]:


print(vars(train_data.examples[0]))


# Our next step is to build the tag vocabulary so they can be numericalized during training. We do this by using the field's `.build_vocab` method on the `train_data`.

# In[298]:


UD_TAGS.build_vocab(train_data)

print(UD_TAGS.vocab.stoi)


# Next, we'll define our iterators. This will define how batches of data are provided when training. We set a batch size and define `device`, which will automatically put our batch on to the GPU, if we have one.
# 
# The BERT model is quite large, so the batch size here is usually smaller than usual. However, the BERT paper itself mentions how they also fine-tuned using small batch sizes, so this shouldn't cause too much of an issue.

# In[299]:


def sort_key(example):
    return len(example.text)

BATCH_SIZE = 32
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device,
    sort_key = sort_key)


# ## Building the Model
# 
# Next up is defining our model. The model is relatively simple, with all of the complicated parts contained inside the BERT module which we do not have to worry about. We can think of the BERT as an embedding layer and all we do is add a linear layer on top of these embeddings to predict the tag for each token in the input sequence. 
# 
# ![](https://github.com/bentrevett/pytorch-pos-tagging/blob/master/assets/pos-bert.png?raw=1)
# 
# Previously the yellow squares were the embeddings provided by the embedding layer, but now they are embeddings provided by the pretrained BERT model. All inputs are passed to BERT at the same time. The arrows between the BERT embeddings indicate how BERT does not calculate embeddings for each tokens individually, but the embeddings are actually based off the other tokens within the sequence. We say the embeddings are *contextualized*.
# 
# One thing to note is that we do not define an `embedding_dim` for our model, it is the size of the output of the pretrained BERT model and we cannot change it. Thus, we simply get the `embedding_dim` from the model's `hidden_size` attribute.
# 
# BERT also wants sequences with the batch element first, hence we permute our input sequence before passing it to BERT.

# In[300]:


class BERTPoSTagger(nn.Module):
    def __init__(self,
                 bert,
                 output_dim, 
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.fc = nn.Linear(embedding_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
  
        #text = [sent len, batch size]
    
        text = text.permute(1, 0)
        
        #text = [batch size, sent len]
        
        embedded = self.dropout(self.bert(text)[0])
        
        #embedded = [batch size, seq len, emb dim]
                
        embedded = embedded.permute(1, 0, 2)
                    
        #embedded = [sent len, batch size, emb dim]
        
        predictions = self.fc(self.dropout(embedded))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions


# Next, we load the actual pretrained BERT uncased model - before we only loaded the tokenizer associated with the model.
# 
# The first time we run this it will have to download the pretrained parameters.

# In[301]:


bert = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')


# ## Training the Model
# 
# We finally get to instantiate our model - a simple linear model using BERT model to get word embeddings.
# 
# Best of all, the only hyperparameter is dropout! This value has been chosen as it's a sensibile value, so there may be a better value of dropout available.

# In[302]:


OUTPUT_DIM = len(UD_TAGS.vocab)
DROPOUT = 0.25

model = BERTPoSTagger(bert,
                      OUTPUT_DIM, 
                      DROPOUT)


# We can then count the number of trainable parameters. This includes the linear layer and all of the BERT parameters.

# In[303]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('The model has {:,} trainable parameters'.format(count_parameters(model)))


# Next, we define our optimizer. Usually when fine-tuning you want to use a lower learning rate than normal, this is because we don't want to drastically change the parameters as it may cause our model to forget what it has learned. This phenomenon is called catastrophic forgetting.
# 
# We pick 5e-5 (0.00005) as it is one of the three values recommended in the BERT paper. Again, there may be better values for this dataset.

# In[304]:


LEARNING_RATE = 5e-5

optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)


# The rest of the notebook is pretty similar to before.
# 
# We define a loss function, making sure to ignore losses whenever the target tag is a padding token.

# In[305]:


TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)


# Then, we place the model on to the GPU, if we have one.

# In[306]:


model = model.to(device)
criterion = criterion.to(device)


# Like in the previous tutorial, we define a function which calculates our accuracy of predicting tags, ignoring predictions over padding tokens.

# In[307]:


def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)


# We then define our `train` and `evaluate` functions to train and test our model. 

# In[308]:


def train(model, iterator, optimizer, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        try:
        
            text = batch.text
            tags = batch.udtags
                    
            optimizer.zero_grad()
            
            #text = [sent len, batch size]
            
            predictions = model(text)
            
            #predictions = [sent len, batch size, output dim]
            #tags = [sent len, batch size]
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            #predictions = [sent len * batch size, output dim]
            #tags = [sent len * batch size]
            
            loss = criterion(predictions, tags)
                    
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        except KeyError:
            continue
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[309]:


def evaluate(model, iterator, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            
            try:
                text = batch.text
                tags = batch.udtags
                
                predictions = model(text)
                
                predictions = predictions.view(-1, predictions.shape[-1])
                tags = tags.view(-1)
                
                loss = criterion(predictions, tags)
                
                acc = categorical_accuracy(predictions, tags, tag_pad_idx)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
            except KeyError:
                continue
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Then, we define a helper function used to see how long an epoch takes.

# In[310]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Finally, we can train our model!
# 
# This model takes a considerable amount of time per epoch compared to the last model as the number of parameters is significantly higher. However, we beat the performance of our last model after only 2 epochs which takes around 2 minutes.

# In[311]:


model_path = 'models/pos-tagging-model.pt'

N_EPOCHS = 10
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_path)
    
    print('Epoch: %02d | Epoch Time: %dm %ds' % (epoch+1, epoch_mins, epoch_secs))
    print('\tTrain Loss: %.3f | Train Acc: %.2f%%' % (train_loss, train_acc*100))
    print('\t Val. Loss: %.3f |  Val. Acc: %.2f%%' % (valid_loss, valid_acc*100))


# We can then load our "best" performing model and try it out on the test set. 
# 
# We beat our previous model by 2%!

# In[ ]:


model.load_state_dict(torch.load(model_path))

test_loss, test_acc = evaluate(model, test_iterator, criterion, TAG_PAD_IDX)

print('Test Loss: %.3f | Test Acc: %.2f%%' % (test_loss, test_acc*100))


# ## Inference
# 
# We'll now see how to use our model to tag actual sentences. This is similar to the inference function from the previous notebook with the tokenization changed to match the format of our pretrained model.
# 
# If we pass in a string, this means we need to split it into individual tokens which we do by using the `tokenize` function of the `tokenizer`. Afterwards, numericalize our tokens the same way we did before, using `convert_tokens_to_ids`. Then, we add the `[CLS]` token index to the beginning of the sequence. 
# 
# **Note**: if we forget to add the `[CLS]` token our results will not be good!
# 
# We then pass the text sequence through our model to get a prediction for each token and then slice off the predictions for the `[CLS]` token as we do not care about it.

# In[ ]:


def tag_sentence(model, device, sentence, tokenizer, text_field, tag_field):
    
    model.eval()
    
    if isinstance(sentence, str):
        tokens = tokenizer.tokenize(sentence)
    else:
        tokens = sentence
    
    numericalized_tokens = tokenizer.convert_tokens_to_ids(tokens)
    numericalized_tokens = [text_field.init_token] + numericalized_tokens
        
    unk_idx = text_field.unk_token
    
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    
    token_tensor = torch.LongTensor(numericalized_tokens)
    
    token_tensor = token_tensor.unsqueeze(-1).to(device)
         
    predictions = model(token_tensor)
    
    top_predictions = predictions.argmax(-1)
    
    predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]
    
    predicted_tags = predicted_tags[1:]
        
    assert len(tokens) == len(predicted_tags)
    
    return tokens, predicted_tags, unks


# We can then run an example sentence through our model and receive the predicted tags.

# In[ ]:


sentence = 'A rainha vai dar um discusso sobre o conflito na Síria amanhã.'

tokens, tags, unks = tag_sentence(model, 
                                  device, 
                                  sentence,
                                  tokenizer,
                                  TEXT, 
                                  UD_TAGS)

print(unks)


# We can then print out the tokens and their corresponding tags.
# 
# Notice how "1pm" in the input sequence has been converted to the two tokens "1" and "##pm". What's with the two hash symbols in front of the "pm"? This is due to the way the tokenizer tokenizes sentences. It uses something called [byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) to split words up into more common subsequences of characters.

# In[ ]:


print("Pred. Tag\tToken\n")

for token, tag in zip(tokens, tags):
    print("%s\t\t%s" % (tag, token))


# We've now fine-tuned a BERT model for part-of-speech tagging! Well done us!
