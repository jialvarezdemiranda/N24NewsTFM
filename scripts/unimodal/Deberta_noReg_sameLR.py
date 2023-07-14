import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger


if __name__== '__main__': # For potential concurrency issues with dataloaders

    SEED=1234542

    pl.seed_everything(SEED, workers=True)

    df_train=pd.read_csv('../../data/splitted/train.csv')
    df_validation=pd.read_csv('../../data/splitted/validation.csv')
    df_test=pd.read_csv('../../data/splitted/test.csv')

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
            self.activation1 = nn.GELU()
            self.output = nn.Linear(512, NUM_CLASSES)

            
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

    experiment_name=f'Deberta_{MAX_LENGTH}_NoRegSameLR' 
    # Define the callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='../../model_ckpts/Unimodal/Text/Deberta',
        filename=experiment_name,
        monitor='val_f1', mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping('val_f1', patience=15,mode='max')

    # instantiate the logger object
    logger = CSVLogger(save_dir="../../logs/Unimodal/Text/Deberta", name=experiment_name)
    

    my_model=TextClassifier(pretrained_model)
    trainer=pl.Trainer(accelerator="gpu", devices=[1], deterministic=True, max_epochs=60, logger=logger, precision='16-mixed', accumulate_grad_batches=2, 
                    callbacks=[lr_monitor, early_stopping, checkpoint_callback])
    trainer.fit(model=my_model,train_dataloaders=train_loader, val_dataloaders=validation_loader)