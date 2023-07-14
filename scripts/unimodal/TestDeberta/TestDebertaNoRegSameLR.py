import argparse
import sys

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel,AutoConfig, DataCollatorWithPadding
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
from PIL import Image

import os
import torchvision.transforms as T
from torchvision.transforms import Compose
from datasets import Dataset, DatasetDict
import seaborn as sns

# Import the file in which the class is located



def main(checkpoint):
    
    SEED=1234542

    pl.seed_everything(SEED, workers=True)

    df_train=pd.read_csv('../../../data/splitted/train.csv')
    df_validation=pd.read_csv('../../../data/splitted/validation.csv')
    df_test=pd.read_csv('../../../data/splitted/test.csv')

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(df_train)
    dataset['validation'] = Dataset.from_pandas(df_validation)
    dataset['test'] = Dataset.from_pandas(df_test)

    NUM_CLASSES= len(df_train['labels'].unique())
    TEXT_USED='text_no_cap'
    # MAX_LENGTH=4096
    # MAX_LENGTH=4096
    MAX_LENGTH=512

    # Load transformer and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #MODEL_NAME = 'microsoft/deberta-v3-base' # 512 seq length
    # MODEL_NAME = 'allenai/longformer-base-4096' # 4096 seq length
    # MODEL_NAME = 'mnaylor/mega-base-wikitext' # 2048 seq length
    MODEL_NAME='microsoft/deberta-base'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    pretrained_model = AutoModel.from_pretrained(MODEL_NAME)

    # Tokenize text
    def tokenize(batch):
        tokens = tokenizer(batch[TEXT_USED], truncation=True, max_length=MAX_LENGTH)
        batch['input_ids'], batch['attention_mask'] = tokens['input_ids'], tokens['attention_mask']
        return batch

    dataset = dataset.map(tokenize)

    dataset['train'] = dataset['train'].remove_columns(['headline', 'abstract', 'caption', 'image_url', 'article_url', 'image_id', 'body', 'full_text', 'text_no_cap', 'labels_text'])
    dataset['validation'] = dataset['validation'].remove_columns(['headline', 'abstract', 'caption', 'image_url', 'article_url', 'image_id', 'body', 'full_text', 'text_no_cap', 'labels_text'])
    dataset['test'] = dataset['test'].remove_columns(['headline', 'abstract', 'caption', 'image_url', 'article_url', 'image_id', 'body', 'full_text', 'text_no_cap', 'labels_text'])

    # Define data loaders

    BATCH_SIZE = 8

    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, collate_fn=data_collator,shuffle=True, num_workers=4, prefetch_factor=8, pin_memory=True)
    validation_loader = DataLoader(dataset['validation'], batch_size=BATCH_SIZE, collate_fn=data_collator, num_workers=4, prefetch_factor=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, collate_fn=data_collator, num_workers=4, prefetch_factor=8, pin_memory=True)

    # Define the model architecture

    class TextClassifier(pl.LightningModule):
        def __init__(self, model=pretrained_model,  lr_transformer=2e-5):
            super(TextClassifier, self).__init__()
            self.criterion = nn.CrossEntropyLoss()
            self.lr_transformer=lr_transformer
            
            # En el train hacemos media de medias
            self.train_loss=[]
            self.train_accs=[]
            self.train_f1s=[]
            
            
            # Aqui computamos las métricas con todo para mayor precision   
            self.val_loss=[]             
            self.all_val_y_true=[]
            self.all_val_y_pred=[]
            
            self.model = model
            self.fc1 = nn.Linear(config.hidden_size, 512)
            # self.fc1 = nn.Linear(config.hidden_size, 64) # Mega
            self.activation1 = nn.GELU()
            self.output = nn.Linear(512, NUM_CLASSES)
            # self.output = nn.Linear(64, NUM_CLASSES) # Mega
            
        def compute_outputs(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['last_hidden_state'][:, 0]  #Get the CLS tokens (deberta)
            x = self.activation1(self.fc1(logits))
            return self.output(x)
        
        def forward(self, batch):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            x = self.compute_outputs(input_ids, attention_mask)
            return x
        
        def training_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            #Compute the output logits
            logits = self.compute_outputs(input_ids, attention_mask)
            #Compute metrics
            loss=self.criterion(logits,labels)
            preds = torch.argmax(logits, dim=-1)
            acc=accuracy_score(y_true=labels.tolist(), y_pred=preds.tolist())
            f1=f1_score(y_true=labels.tolist(), y_pred=preds.tolist(), average='macro')
            self.train_loss.append(loss)
            self.train_accs.append(acc)
            self.train_f1s.append(f1)
            
            return loss
        
        def on_train_epoch_end(self):
            # outs is a list of whatever you returned in `validation_step`
            mean_loss = sum(self.train_loss)/len(self.train_loss)
            mean_acc=sum(self.train_accs)/len(self.train_accs)
            mean_f1=sum(self.train_f1s)/len(self.train_f1s)
            
            self.log("train_loss", mean_loss)
            self.log("train_acc", mean_acc)
            self.log("train_f1", mean_f1)
            
            self.train_loss=[]
            self.train_accs=[]
            self.train_f1s=[]
        
        
        def validation_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            #Compute the output logits
            logits = self.compute_outputs(input_ids, attention_mask)
            #Compute metrics
            loss=self.criterion(logits,labels)
            preds = torch.argmax(logits, dim=-1)
            
            self.val_loss.append(loss)
            
            self.all_val_y_true.extend(labels.tolist())
            self.all_val_y_pred.extend(preds.tolist())
            return loss
        
        def on_validation_epoch_end(self):
            # outs is a list of whatever you returned in `validation_step`
            mean_loss = sum(self.val_loss)/len(self.val_loss)
            
            acc= accuracy_score(y_true=self.all_val_y_true, y_pred=self.all_val_y_pred)
            f1= f1_score(y_true=self.all_val_y_true, y_pred=self.all_val_y_pred, average='macro')
            
            self.log("val_loss", mean_loss)
            self.log("val_acc", acc)
            self.log("val_f1", f1)
            
            self.val_loss=[]
            self.all_val_y_true=[]
            self.all_val_y_pred=[]
        
        def configure_optimizers(self):
            optimizer = optim.AdamW([
                {'params': self.parameters()},
            ],lr=self.lr_transformer, amsgrad=True, weight_decay=0.01)
            
            scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=5)
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        },
                    }

    checkpoint_path= '/home/nacho/work/model_ckpts/Unimodal/Text/Deberta/Deberta_512_NoRegSameLR.ckpt'
    # Load the model from the checkpoint
    model = TextClassifier.load_from_checkpoint(checkpoint_path, map_location='cuda:2')

    #Get predictions
    trainer=pl.Trainer(accelerator="gpu", devices=[2], deterministic=True, max_epochs=25, precision=16, accumulate_grad_batches=2)
    predictions_test = trainer.predict(model, test_loader)
    predictions_val = trainer.predict(model, validation_loader)
    
    compute_metrics(validation_loader, predictions_val, 'val', checkpoint)
    compute_metrics(test_loader, predictions_test, 'test', checkpoint)
    
    


def compute_metrics(loader, predictions, split, ckpt):
    original_stdout = sys.stdout # Save a reference to the original standard output
    # Redirect print statements to a file
    sys.stdout = open(f'../../../Val-TestResults/Unimodal/Deberta/Metrics/{ckpt}.txt', 'a')
    
    # initialize the variables for storing true and predicted labels
    all_y_true = []
    all_y_pred = []

    # iterate over the batches and compute f1-score and confusion matrix for each batch
    for i, batch in enumerate(loader):
        preds=torch.argmax(predictions[i], dim=-1)
        y_pred, y_true = preds.tolist(), batch['labels'].tolist()

        # append the true and predicted labels to the corresponding lists
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

      
    mean_f1= f1_score(y_true=all_y_true, y_pred=all_y_pred, average='macro')
    mean_acc=accuracy_score(y_true=all_y_true, y_pred=all_y_pred)
    print(f'{split} Acc: {mean_acc}')
    print(f'{split} F1: {mean_f1}')
    print('-----------')
    
    sys.stdout = original_stdout # Reset the standard output to its original value
    
    cfm(all_y_true,all_y_pred, split, ckpt)
    
def cfm(y_true, y_pred, split, ckpt):

    CLASSES_DICT= {0: 'Movies', 1: 'Sports', 2: 'Music', 3: 'Opinion', 4: 'Media', 5: 'Art & Design', 6: 'Theater', 7: 'Television', 8: 'Technology', 9: 'Economy', 10: 'Books', 11: 'Style', 12: 'Travel', 13: 'Health', 14: 'Real Estate', 15: 'Dance', 16: 'Science', 17: 'Fashion', 18: 'Well', 19: 'Food', 20: 'Your Money', 21: 'Education', 22: 'Automobiles', 23: 'Global Business'}
    # compute the confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # plot the confusion matrix as a heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')

    # set the plot labels
    class_names = [CLASSES_DICT[i] for i in CLASSES_DICT]
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(np.arange(24) + 0.5, class_names, rotation=90)
    plt.yticks(np.arange(24) + 0.5, class_names, rotation=0)

    dir= f'../../../Val-TestResults/Unimodal/Deberta/CFM/{ckpt}'
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.savefig(f'../../../Val-TestResults/Unimodal/Deberta/CFM/{ckpt}/{split}.png')  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get validation and test F1 and accuracy and confusion matrix')
    parser.add_argument('--ckpt', help='Name of the checkpoint file')
    args = parser.parse_args()

    # Call the main function with the arguments
    
    
    main(args.ckpt)
