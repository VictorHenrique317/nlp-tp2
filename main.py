#!/usr/bin/env python
# coding: utf-8

# # NLP TP2 - Victor Henrique Silva Ribeiro
# 
# 
# ## Introdução
# Nesse trabalho, irei utilizar o modelo pré-treinado `bert-base-portuguese-cased` para a tarefa downstream de POS tagging. Para isso, ultilizo o dataset `macmorpho`, que é um dataset de POS tagging para o português.
# 
# Primeiramente importo as bibliotecas necessárias para o trabalho.

# In[35]:


get_ipython().run_line_magic('pip', 'install torchtext==0.6.0')

import torch

import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext.data import Example, Dataset

from transformers import BertTokenizer, BertModel

import numpy as np

import random
import functools

from datasets import load_dataset

import time


# O primeiro passo é importar o tokenizador em português utilizando a biblioteca `transformers` do HuggingFace. É importante lembrar que é necessário utilizar em nossos inputs os tokens de começo de frase, token desconhecido e padding que foram utilizados no treinamento do `BERT`. Além disso precisamos truncar nossos inputs para o tamanho máximo de tokens que o `BERT` suporta, que é 512.

# In[6]:


tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

init_token = tokenizer.cls_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']


# Agora definimos como os inputs e labels serão pré-processados para o formato que o `BERT` espera. Todo o processo é feito utilizando tensores `PyTorch`.

# In[10]:


def inputProcessor(tokens, tokenizer, max_input_length):
    tokens = tokens[:max_input_length-1]
    tokens = [tokenizer.convert_tokens_to_ids(token) 
              if token in tokenizer.vocab 
              else tokenizer.convert_tokens_to_ids('<unk>') 
              for token in tokens]
    return tokens

def labelProcessor(tokens, max_input_length):
    tokens = tokens[:max_input_length-1]
    return tokens

text_preprocessor = functools.partial(inputProcessor,
                                      tokenizer = tokenizer,
                                      max_input_length = max_input_length)

tag_preprocessor = functools.partial(labelProcessor,
                                     max_input_length = max_input_length)

TEXT = data.Field(use_vocab = False,
                  lower = True,
                  preprocessing = text_preprocessor,
                  init_token = init_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

UD_TAGS = data.Field(unk_token = None,
                     init_token = '<pad>',
                     preprocessing = tag_preprocessor)

fields = (("tokens", TEXT), ("pos_tags", UD_TAGS))


# Aqui importamos o dataset `macmorpho` utilizando a biblioteca `datasets` do HuggingFace. O dataset é dividido em treino, validação e teste. O dataset de treino é utilizado para treinar o modelo, o de validação é utilizado para escolher o melhor modelo e o de teste é utilizado para avaliar o modelo final.
# 
# Depois de definidas as divisões transformo elas em tensores `PyTorch` usando os procedimentos definidos anteriormente.

# In[15]:


def toPytorchDataset(dataset, train_set=None):
    dataset = [(example['tokens'], example['pos_tags']) for example in dataset]

    examples = [Example.fromlist([text, tags], fields=[('text', TEXT), ('udtags', UD_TAGS)]) for text, tags in dataset]
    dataset = Dataset(examples, fields=[('text', TEXT), ('udtags', UD_TAGS)])

    return dataset


dataset = load_dataset('mac_morpho')
train_data_raw = dataset['train']
valid_data_raw = dataset['validation']
test_data_raw = dataset['test']

train_data = toPytorchDataset(train_data_raw)
valid_data = toPytorchDataset(valid_data_raw, train_set=train_data_raw)
test_data = toPytorchDataset(test_data_raw, train_set=train_data_raw)

print(len(train_data.examples))
print(len(valid_data.examples))
print(len(test_data.examples))


# É necessário construir o vocabulário para as tags, para que elas possam ser indexadas durante o treinamento.

# In[17]:


UD_TAGS.build_vocab(train_data)
print(UD_TAGS.vocab.stoi)


# Importando o modelo pré-treinado `bert-base-portuguese-cased` e adicionando a camada linear no final para classificar as tags.

# In[19]:


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
        text = text.permute(1, 0)

        embedded = self.dropout(self.bert(text)[0])
        embedded = embedded.permute(1, 0, 2)

        predictions = self.fc(self.dropout(embedded))
        
        return predictions
    
bert = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

OUTPUT_DIM = len(UD_TAGS.vocab)
DROPOUT = 0.25

model = BERTPoSTagger(bert,
                      OUTPUT_DIM, 
                      DROPOUT)


# Agora defino o procedimento de treinamento da camada linear. Todo o processo será realizado na CPU.

# In[18]:


def sort_key(example):
    return len(example.text)

BATCH_SIZE = 32
device = torch.device('cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device,
    sort_key = sort_key)

LEARNING_RATE = 5e-5
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

model = model.to(device)
criterion = criterion.to(device)


# Definindo as funções de treino e avaliação do modelo.

# In[27]:


def getAccuracy(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim = 1, keepdim = True)
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)

def train(model, iterator, optimizer, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        try:
            text = batch.text
            tags = batch.udtags
                    
            optimizer.zero_grad()
            
            predictions = model(text)
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)
            acc = getAccuracy(predictions, tags, tag_pad_idx)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        except KeyError:
            continue
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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
                
                acc = getAccuracy(predictions, tags, tag_pad_idx)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
            except KeyError:
                continue
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Treinando o modelo.

# In[30]:


model_path = 'models/pos-tagging-model.pt'

N_EPOCHS = 10
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_path)
    
    print('Epoch: %02d' % (epoch+1))
    print('\tTrain Loss: %.3f | Train Acc: %.2f%%' % (train_loss, train_acc*100))
    print('\t Val. Loss: %.3f |  Val. Acc: %.2f%%' % (valid_loss, valid_acc*100))


# Carregando o modelo com a melhor acurácia no dataset de validação e testando no dataset de teste. Obtendo uma acurácia de 93.29%.

# In[31]:


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

# In[32]:


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

# In[33]:


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

# In[34]:


print("Pred. Tag\tToken\n")

for token, tag in zip(tokens, tags):
    print("%s\t\t%s" % (tag, token))


# We've now fine-tuned a BERT model for part-of-speech tagging! Well done us!

# In[ ]:




