# -*- coding: utf-8 -*-
"""
T5Summarization.ipynb

Original file is located at
    https://colab.research.google.com/drive/1kYVQqol5iIwEye1nm5cXZ27IkIMk1KB1

Creating a summary of the Amazon Food Products Review with the help of the T5 Model

---

Automated summary is a summary of the text created with the help of the computer.
There are two types of such summaries: *abstractive*, which resembles marking down the most crucial parts of the text,
                                        and *extractive*, which can be compared to the human-written summary.

This project shows the implementation of the Google’s text-to-text transfer transformer model (T5)
for creating a summary of the Amazon Food Product Reviews.
"""

# Preinstalling necessary libraries
!pip install --quiet transformers==4.5
!pip install --quiet rouge_score
!pip install --quiet pytorch-lightning
!pip install --quiet tensorflow
!pip install --quiet tensorboard
!pip install --quiet nltk
!pip install --quiet NLP-python

# Importing necessary packages
import pandas as pd
import numpy as np
import logging 
import torch
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from dataclasses import dataclass#, field
from typing import Optional

# Importing the tokenizer and T5 model
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AdamW, Trainer, TrainingArguments

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from rouge_score import rouge_scorer

# Packages and libraries for removing stopwords
import re
from NLP import NLP

# Import NLP
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Importing Google Drive for reading the data
from google.colab import drive
drive.mount('/content/drive')

# Reading and clearing the data
df=pd.read_csv("/data_input_direction", engine="python", error_bad_lines=False)


# Deleting all columns that are not usefull for training of the summarization model
df.drop(columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator','HelpfulnessDenominator', 'Score', 'Time'],axis=1,inplace=True)
print("Before",len(df))
df = df.dropna() #deleting enteries without any value
print("Data size:",len(df))
df.head() #preview of data


# Shortening the data for testing purposes and comparing the shapes of the full and shortened set
# Remove in case the full training is possible
df1=df.loc[1:100000]  
print("Data size:",len(df1))


# Untokenize function from
# https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py

def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

text = df1['Text']


# Convert text to lowercase and split to a list of words
tokens=[]
for i in range(len(text)):
  oneRow=text.iloc[i]
  tokens.append(word_tokenize(oneRow.lower()))


# Initializing english stopwords to remove them from each Text of review
english_stopwords = stopwords.words('english')
tokensWoStopwords=[]


# Going through each tokenized review text to remove all stopwords
for i in range(len(tokens)):
  tokens_wo_stopwords = [t for t in tokens[i] if t not in english_stopwords]
  # Appending untokenized review Text to new list 
  # but only keeping first 512 tokens, due to model limits
  tokensWoStopwords.append(untokenize(tokens_wo_stopwords[:512]))


# Replacing Text with Text without stopwords
# In case of full training, change df1 to df
df1['Text']=tokensWoStopwords
df1=df1.reset_index(drop=True)
df1.head #preview of data


# Shortened dataset split into train, validation and test dataset 
# Used ratio 80/10/10
# Remove in case the full training is possible
n_train = int(np.round(df1.shape[0]*0.8))
n_val = int(np.round(df1.shape[0]*0.1))
n_test = int(np.round(df1.shape[0]*0.1))
train_data=df1.loc[:n_train]
val_data=df1.loc[n_train:n_train+n_val]
test_data=df1.loc[n_train+n_val:n_train+n_val+n_test]


#Full dataset split
# Used ratio 80/10/10
#n_train = int(np.round(df.shape[0]*0.8))
#n_val = int(np.round(df.shape[0]*0.1))
#n_test = int(np.round(df.shape[0]*0.1))
#train_data=df.loc[:n_train]
#val_data=df.loc[n_train:n_train+n_val]
#test_data=df.loc[n_train+n_val:n_train+n_val+n_test]


# Checking shape how dataset is splitted
train_data.shape, test_data.shape, val_data.shape


# Creating dataset shape for the new T5 model for summarisation
class SummaryDataset (Dataset):
  def __init__ (
      self,
      data: pd.DataFrame,
      tokenizer: T5TokenizerFast, #initializing tokenizer
      text_max_token_len: int = 512, #setting maximum lenght of tokens for text
      summary_max_token_len: int = 128 #setting maximum lenght of tokens for summary
      ):
    self.tokenizer = tokenizer
    self.dataF = data 
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len
  
  def __len__(self):
    return len(self.dataF)

  def __getitem__(self, index: int):
    data_row = self.dataF.iloc[index]
    
    text = data_row["Text"]
    #encoding Text value to be suitable for pretrained T5 model
    text_encoding = tokenizer( 
        text,
        max_length=self.text_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"

    )
    #encoding Summary value to be suitable for pretrained T5 model
    summary_encoding = tokenizer(
        data_row["Summary"],
        max_length=self.summary_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"

    )

    labels = summary_encoding["input_ids"]
    labels[labels==0]=-100
    #reurning dictionary of tokenized Text and Summary of reviews 
    return dict(
        text=text,
        summary=data_row["Summary"],
        text_input_ids=text_encoding["input_ids"].flatten(),
        text_attention_mask=text_encoding["attention_mask"].flatten(),
        labels=labels.flatten(),
        labels_attention_mask=summary_encoding["attention_mask"].flatten()
    )

# Encoding train, validation and test dataset to desired input of the model 
class SummaryDataModule(pl.LightningDataModule):
  def __init__(
    self,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: T5TokenizerFast,
    batch_size: int = 8,
    text_max_token_len: int = 512,
    summary_max_token_len: int = 128
  ):
      
    super ().__init__()
      
    self.train_df = train_df
    self.test_df = test_df
    self.val_df=val_df

    self.batch_size = batch_size
    self.tokenizer = tokenizer
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len
    
  def setup(self, stage=None) :
    self.train_dataset = SummaryDataset(
        self.train_df,
        self.tokenizer,
        self.text_max_token_len,
        self.summary_max_token_len
        )
   
    
    self.test_dataset = SummaryDataset(
        self.test_df,
        self.tokenizer,
        self.text_max_token_len,
        self.summary_max_token_len
        )
   
    self.val_dataset = SummaryDataset(
        self.val_df,
        self.tokenizer,
        self.text_max_token_len,
        self.summary_max_token_len
        )

    
  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=2
        )
 
  def val_dataloader(self):
   return DataLoader(
        self.val_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2
        )
  def test_dataloader(self):
   return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2
        )


# Initialising tokenizer 
modelName="t5-small" 
tokenizer = T5TokenizerFast.from_pretrained(modelName)

text_token_counts =[] 
summary_token_counts = []


# Checking distribution of tokens in columns Text and Summary to get feeling about data distribution 
for _,row in train_data.iterrows():
  text_token_count = len(tokenizer.encode(row["Text"][:512])) #keeping first 512 tokenized values
  #NOTE this lenght is not the same as lenght of Text of review
  text_token_counts.append(text_token_count)

  summary_token_count = len(tokenizer.encode(row["Summary"]))
  summary_token_counts.append(summary_token_count)

# Plotting lenght of text and summaries to see how many tokens we have each
fig, (ax1, ax2) = plt.subplots(1, 2)

sns.histplot(text_token_counts, ax=ax1)
ax1.set_title('full text token counts')

sns.histplot(summary_token_counts, ax=ax2)
ax2.set_title('summary text token counts')

# Initializing the rouge metrics and defining the training parameters
!pip install --quiet datasets==1.0.

# Training parameters set up
N_EPOCHS = 5 # number of epochs which the training will go for
TRAIN_BATCH_SIZE = 16 # defines number of samples that will be propagated
BATCH_SIZE = 16


# Initializing data_module with train, test, validation dataset and tokenizer
data_module=SummaryDataModule(train_data,test_data,val_data,tokenizer,batch_size=BATCH_SIZE)


# Creating T5 model for summarization
class SummaryModel(pl.LightningModule):

 def __init__(self):
   super().__init__()
   #initializing model
   self.model = T5ForConditionalGeneration.from_pretrained(modelName, return_dict=True) 
   
        
 #defining forward function and it output
 def forward(self,input_ids, attention_mask, decoder_attention_mask, labels=None):
   output = self.model(
      input_ids,
      attention_mask=attention_mask,
      labels=labels,
      decoder_attention_mask=decoder_attention_mask
    )
   return output.loss, output.logits

 
 def training_step(self, batch, batch_idx):
   input_ids = batch[ "text_input_ids"]
   attention_mask = batch["text_attention_mask"]
   labels = batch["labels"]
   x = batch[ "text_input_ids"]
   labels_attention_mask = batch["labels_attention_mask"]
  
   loss, outputs = self(
     input_ids=input_ids,
     attention_mask=attention_mask,
     decoder_attention_mask=labels_attention_mask,
     labels=labels
    )
   
   #pred=self.forward(x, attention_mask, labels_attention_mask, labels)
   #correct=outputs.argmax(dim=1).eq(labels).sum().item()
   #total=len(labels)
   
   batch_dictionary={ "loss": loss, "labels": labels}
       
    
   self.log("Loss/Train (Batch)", loss, prog_bar=True,logger=True)
   self.logger.experiment.add_scalar("Loss/Train (Epoch)", loss, self.current_epoch)
   #return loss
   return batch_dictionary

 # The following function allows the output of the training parameters after each epoch
 # Disabled due to Tensorboard logging set up, can be used in case when other output strategy is needed
 '''
 def training_epoch_end(self,outputs):
   
   avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
   correct=sum([x["correct"] for  x in outputs])
   total=sum([x["total"] for  x in outputs])

   self.logger.experiment.add_scalar("Train Loss",avg_loss,self.current_epoch)
   self.logger.experiment.add_scalar("Train Accuracy",correct/total,self.current_epoch)

   epoch_dictionary={'loss': avg_loss}

   return epoch_dictionary
 '''
 
 def validation_step(self, batch, batch_idx):
   input_ids = batch[ "text_input_ids"]
   attention_mask = batch["text_attention_mask"]
   labels = batch["labels"]
   labels_attention_mask = batch["labels_attention_mask"]
  
   loss, outputs = self(
     input_ids=input_ids,
     attention_mask=attention_mask,
     decoder_attention_mask=labels_attention_mask,
     labels=labels
    )
   
   
   self.logger.experiment.add_scalar("Loss/Val (epoch)",loss,self.current_epoch)
  
   self.log("Loss/Val (Batch)", loss, prog_bar=True,logger=True)

   epoch_dictionary={'loss': loss
                     }
   return epoch_dictionary

 # The following function allows the output of the validation parameters after each epoch
 # Disabled due to Tensorboard logging set up, can be used in case when other output strategy is needed
 '''
 def validation_epoch_end(self, outputs):
        #outputs = str(outputs)
        
        #avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        #tensorboard_logs = {"val_loss": avg_loss}
        
        rouge_results = self.rouge_metric.compute() 
        rouge_dict = self.parse_score(rouge_results)
    
        #result = metric.compute(predictions=outputs, references=labels, use_stemmer=True)
      
        #tensorboard_logs = {"Rougue1": rouge_results['rouge1']}
        #tensorboard_logs.update(rouge1=rouge_dict['rouge1'], rougeL=rouge_dict['rougeL'])
        ## Clear out the lists for next epoch
        self.target_gen= []
        self.prediction_gen=[]
        #self.log("rougue1", rouge_results['rouge1'], prog_bar=True,logger=True)
        #self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        print('Rogue results',rouge_results['rouge1'])
        self.logger.experiment.add_scalar("Rogue1",rouge_results['rouge1'],self.current_epoch)
        self.logger.experiment.add_scalar("RogueL",rouge_results['rougeL'],self.current_epoch)
        rougex = rouge_results['rouge1']
        epoch_dictionary={'loss': avg_loss, 
                         "rouge1" : rouge_results['rouge1'],
                         "rougeL" : rouge_results['rougeL']}


        return epoch_dictionary
 '''

 def test_step(self, batch, batch_idx):
   input_ids = batch[ "text_input_ids"]
   attention_mask = batch["text_attention_mask"]
   labels = batch["labels"]
   labels_attention_mask = batch["labels_attention_mask"]
  
   loss, outputs = self(
     attention_mask=attention_mask,
     decoder_attention_mask=labels_attention_mask,
     labels=labels
    )
   self.logger.experiment.add_scalar("Loss/Test",loss,self.current_epoch)
   self.log("test_loss", loss, prog_bar=True,logger=True)
   return {'loss': loss}

# Configurating optimizer as most used one AdamW
 def configure_optimizers(self):
    return AdamW(self.parameters(), lr=0.0001)


model=SummaryModel()

# Creating checkpoint_callback for the model
# and setting which function we are monitoring
from pytorch_lightning.profiler import SimpleProfiler

profiler = SimpleProfiler()

checkpoint_callback = ModelCheckpoint(
  dirpath="checkpoints", 
  filename="best-checkpoint",
  save_top_k=1,
  verbose=True,
  monitor="val_loss",
  mode="min"
)

# Initializing the logger
logger = TensorBoardLogger("lightning_logs", name="T5-summary")

# Initializing trainer
trainer = pl.Trainer(
  logger=logger,
  enable_checkpointing=checkpoint_callback,
  profiler = profiler,
  max_epochs=N_EPOCHS,
  gpus=1,
  enable_progress_bar=True
)

trainer.fit(model,data_module) #training the model

# Saving the trained model 
trained_model = SummaryModel.load_from_checkpoint(
  trainer.checkpoint_callback.best_model_path
)
trained_model.freeze()


# Commented out IPython magic to ensure Python compatibility.
# Loading tensorboard for results output
# %load_ext tensorboard

# Clearing any logs from previous runs (if needed)
#rm -rf ./logs/

# %tensorboard --logdir ./lightning_logs


# Defining function which will generate summaries 
def summarize (text):
  text_encoding = tokenizer(
    text,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt"
  )
  generated_ids = trained_model.model.generate(
    input_ids=text_encoding["input_ids"],
    attention_mask=text_encoding["attention_mask"],
    max_length=200,
    num_beams=2,
    repetition_penalty=2.5,
    length_penalty=1.0,
    early_stopping=True
  ) 
  
  preds =[
    tokenizer.decode (gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    for gen_id in generated_ids
  ]
  
  return "". join(preds)


# Creating new dataset with old summary and new generated summary 
test_data = test_data.reset_index()
for i in range(0,len(test_data)):
  test_data['Generated_summary'] = ""
  #test_data['Rogue'] = ''
test_data.head()


#Generate summary for the test data
for i in range (10): #replace with len(test_data)
  sample_row = test_data.iloc[i]
  text = sample_row["Text"]
  model_summary = summarize(text)
  test_data["Generated_summary"][i] = model_summary
  #scores = scorer.score(text,model_summary)
  #test_data['Rogue'][i] = scorer.score(text,model_summary)


#Calculationg the rogue score for the entire test dataset
import nlp
nlp_rouge = nlp.load_metric('rouge')

scores = nlp_rouge.compute(
    test_data.Generated_summary.to_list(), test_data.Text.to_list(),
    rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
    use_agregator=True, use_stemmer=False
)

df_rouge = pd.DataFrame({
    'rouge1': [scores['rouge1'].mid.precision, scores['rouge1'].mid.recall, scores['rouge1'].mid.fmeasure],
    'rouge2': [scores['rouge2'].mid.precision, scores['rouge2'].mid.recall, scores['rouge2'].mid.fmeasure],
    'rougeL': [scores['rougeL'].mid.precision, scores['rougeL'].mid.recall, scores['rougeL'].mid.fmeasure]}, index=[ 'P', 'R', 'F'])

df_rouge.style.format({'rouge1': "{:.4f}", 'rouge2': "{:.4f}", 'rougeL': "{:.4f}"})

test_data.head() #preview of data


# Saving generated summaries as new csv file 
test_data.to_csv("/content/drive/MyDrive/summary_test.csv")

sample_row = test_data.iloc[77]
text = sample_row["Text" ]
model_summary = summarize(text)

summary = sample_row["Summary"]
print(summary)
print(model_summary)


# Calculate and print out rouge scores for a particular row
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(model_summary,text)
scores