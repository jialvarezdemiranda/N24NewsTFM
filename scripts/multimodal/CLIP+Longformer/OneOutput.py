
import pandas as pd
import numpy as np
from transformers import AutoModel, CLIPProcessor, AutoConfig, AutoTokenizer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import os
import torchvision.transforms as T
from torchvision.transforms import Compose

from datasets import Dataset, DatasetDict

if __name__== '__main__': # For potential concurrency issues with dataloaders

    SEED=1234542

    pl.seed_everything(SEED, workers=True)

    df_train=pd.read_csv('../../../data/splitted/train.csv')
    df_validation=pd.read_csv('../../../data/splitted/validation.csv')
    df_test=pd.read_csv('../../../data/splitted/test.csv')

    # Remove nan from caption column
    df_train.fillna(value="", inplace=True)
    df_validation.fillna(value="", inplace=True)
    df_test.fillna(value="", inplace=True)

    label_dict={0: 'Movies', 1: 'Sports', 2: 'Music', 3: 'Opinion', 4: 'Media', 5: 'Art & Design', 6: 'Theater', 7: 'Television', 8: 'Technology', 9: 'Economy', 10: 'Books', 11: 'Style', 12: 'Travel', 13: 'Health', 14: 'Real Estate', 15: 'Dance', 16: 'Science', 17: 'Fashion', 18: 'Well', 19: 'Food', 20: 'Your Money', 21: 'Education', 22: 'Automobiles', 23: 'Global Business'}

    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(df_train)
    dataset['validation'] = Dataset.from_pandas(df_validation)
    dataset['test'] = Dataset.from_pandas(df_test)

    NUM_CLASSES= len(df_train['labels'].unique())

    TEXT_CLIP='caption'

    TRAIN_IMAGES_PATH= '../../../images/train'
    VALIDATION_IMAGES_PATH= '../../../images/validation'
    TEST_IMAGES_PATH= '../../../images/test'

    TEXT_TRANSF='text_no_cap'
    MAX_LENGTH=1024


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CLIP_NAME = 'openai/clip-vit-base-patch32'
    TRANSFORMER_NAME= 'allenai/longformer-base-4096'

    clip_model = AutoModel.from_pretrained(CLIP_NAME)
    clip_config= AutoConfig.from_pretrained(CLIP_NAME)
    clip_processor= CLIPProcessor.from_pretrained(CLIP_NAME)

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    config= AutoConfig.from_pretrained(TRANSFORMER_NAME)
    pretrained_model = AutoModel.from_pretrained(TRANSFORMER_NAME)


    def tokenize(batch):
        tokens = tokenizer(batch[TEXT_TRANSF], truncation=True, max_length=MAX_LENGTH)
        batch['input_ids'], batch['attention_mask'] = tokens['input_ids'], tokens['attention_mask']
        return batch

    dataset = dataset.map(tokenize)

    dataset['train'] = dataset['train'].remove_columns(['headline', 'abstract', 'caption', 'image_url', 'article_url', 'image_id', 'body', 'full_text', 'text_no_cap', 'labels_text'])
    dataset['validation'] = dataset['validation'].remove_columns(['headline', 'abstract', 'caption', 'image_url', 'article_url', 'image_id', 'body', 'full_text', 'text_no_cap', 'labels_text'])
    dataset['test'] = dataset['test'].remove_columns(['headline', 'abstract', 'caption', 'image_url', 'article_url', 'image_id', 'body', 'full_text', 'text_no_cap', 'labels_text'])


    class CustomMultimodalDataset(torch.utils.data.Dataset):
        def __init__(self, df, img_dir, ds):
            self.df= df
            self.img_dir = img_dir
            self.ds=ds
            
        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            label_text= self.df['labels_text'].iloc[idx]
            img_path = os.path.join(self.img_dir, label_text, self.df['image_id'].iloc[idx])
            img_path=img_path + '.jpg'
            image = Image.open(img_path)
            if(image.mode != 'RGB'):
                image=image.convert('RGB')
            caption = self.df[TEXT_CLIP].iloc[idx]
            text_input_ids= self.ds[idx]['input_ids']
            text_attention_mask= self.ds[idx]['attention_mask']
            label = self.df['labels'].iloc[idx]
            return image, caption, text_input_ids, text_attention_mask, label
        
    train_dataset= CustomMultimodalDataset(df_train, TRAIN_IMAGES_PATH, dataset['train'])
    validation_dataset= CustomMultimodalDataset(df_validation, VALIDATION_IMAGES_PATH, dataset['validation'])
    test_dataset= CustomMultimodalDataset(df_test, TEST_IMAGES_PATH, dataset['test'])


    class MultimodalCollator:
        HARD_IMG_AUGMENTER = T.RandAugment(num_ops=6, magnitude=9)
        
        def __init__(self, processor=clip_processor, split='train', max_length=77):
            # 40 max length for vilt // 77 max length for clip
            self.processor = processor
            self.split = split
            self.max_length = max_length

        def __call__(self, batch):
            images, captions, text_input_ids, text_attention_masks, labels = list(zip(*batch))
            if self.split=='train':
                images = [self.HARD_IMG_AUGMENTER(img) for img in images]
            
            # Pad text_input_ids and text_attention_masks
            max_length = max(len(ids) for ids in text_input_ids)
            padded_text_input_ids = [ids + [1] * (max_length - len(ids)) for ids in text_input_ids]
            padded_text_attention_masks = [masks + [0] * (max_length - len(masks)) for masks in text_attention_masks]

            encoding = self.processor(images=images, 
                                    text=list(captions), 
                                    padding=True,
                                    max_length=self.max_length,
                                    truncation=True,
                                    return_tensors='pt')
            encoding['text_input_ids'] = torch.tensor(padded_text_input_ids)
            encoding['text_attention_masks'] = torch.tensor(padded_text_attention_masks)
            encoding['labels']=torch.tensor(labels)
            return encoding


    BATCH_SIZE=8

    collator_train=MultimodalCollator(split='train')
    collator_val=MultimodalCollator(split='val')
    collator_test=MultimodalCollator(split='test')
    train_loader = DataLoader(train_dataset, collate_fn=collator_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, prefetch_factor=8, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, collate_fn=collator_val, batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, collate_fn=collator_test, batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=8, pin_memory=True)


    class MultimodalClassifier(pl.LightningModule):
        def __init__(self, clip_model=clip_model, text_transformer= pretrained_model,  lr_transformer=2e-5, lr_heads=2e-3):
            super(MultimodalClassifier, self).__init__()
            self.criterion = nn.CrossEntropyLoss()
            self.lr_transformer=lr_transformer
            self.lr_heads=lr_heads
            # En el train hacemos media de medias
            self.train_loss=[]
            self.train_accs=[]
            self.train_f1s=[]
            
            
            # Aqui computamos las métricas con todo para mayor precision   
            self.val_loss=[]             
            self.all_val_y_true=[]
            self.all_val_y_pred=[]
            
            self.text_transformer = text_transformer
            self.clip_model = clip_model
            # Freeze CLIP model parameters
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # CLIP output
            self.clip_layer_norm = nn.LayerNorm(1024)
            self.clip_fc1 = nn.Linear(1024, 512)
            self.clip_activation1 = nn.GELU()
            self.clip_dropout = nn.Dropout(p=0.2)
        # self.output_clip = nn.Linear(256, NUM_CLASSES)
            
            # Transformer output
            self.transformer_layer_norm = nn.LayerNorm(768)
            self.transformer_fc1 = nn.Linear(768, 512)
            self.transformer_activation1 = nn.GELU()
            self.transformer_dropout = nn.Dropout(p=0.5)
            #self.output_transformer = nn.Linear(512, NUM_CLASSES)
            
            # Se puede probar a concatenar el output de CLIP (512) con una proyección del output del transformer 
            # (768 proyectarlo a 512)
            
            self.final_layer_norm = nn.LayerNorm(1024)            
            self.final_fc1 = nn.Linear(1024, 512)
            self.final_activation1 = nn.GELU()
            self.final_dropout = nn.Dropout(p=0.1)
            self.final_output = nn.Linear(512, NUM_CLASSES)
            
        def compute_outputs(self, input_ids, attention_mask, pixel_values, text_input_ids, text_attention_masks):
            # Get CLIP embedding
            out_text=self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            out_image=self.clip_model.get_image_features(pixel_values=pixel_values)
            
            combined_embed= torch.cat((out_text,out_image), dim=-1) # Concat
            
            x = self.clip_layer_norm(combined_embed)
            
            x=self.clip_fc1(x)
            
            
            clip_embed = self.clip_activation1(x)
            
            clip_embed = self.clip_dropout(clip_embed)
            
            
            # Transformer embedding
            
            outputs = self.text_transformer(input_ids=text_input_ids, attention_mask=text_attention_masks)
            logits = outputs.pooler_output
            
            
            transformer_embed = self.transformer_layer_norm(logits)
            transformer_embed= self.transformer_fc1(transformer_embed)
            
            transformer_embed = self.transformer_activation1(transformer_embed)
            transformer_embed = self.transformer_dropout(transformer_embed)
            
            
            # Combine Transformer and CLIP and get output
            
            final_embed= torch.cat((transformer_embed,clip_embed), dim=-1) # Concat
            final_embed = self.final_layer_norm(final_embed)
            final_embed=self.final_fc1(final_embed)
            z = self.final_activation1(final_embed)
            z = self.final_dropout(z)
            
            return self.final_output(z)
        
        def forward(self, batch):
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            text_input_ids= batch['text_input_ids']
            text_attention_masks=batch['text_attention_masks']
            x = self.compute_outputs(input_ids, attention_mask, pixel_values, text_input_ids, text_attention_masks)
            return x
        
        def training_step(self, batch, batch_idx):
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            text_input_ids= batch['text_input_ids']
            text_attention_masks=batch['text_attention_masks']
            
            labels = batch['labels']
            #Compute the output logits
            logits = self.compute_outputs(input_ids, attention_mask, pixel_values, text_input_ids, text_attention_masks)
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
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            text_input_ids= batch['text_input_ids']
            text_attention_masks=batch['text_attention_masks']
            labels = batch['labels']
            #Compute the output logits
            logits = self.compute_outputs(input_ids, attention_mask, pixel_values, text_input_ids, text_attention_masks)
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
                {'params': self.text_transformer.parameters(), 'lr': self.lr_transformer,'amsgrad':True, 'weight_decay':0.01 },
                {'params': self.clip_fc1.parameters()},
                {'params': self.clip_layer_norm.parameters()},
                {'params': self.transformer_fc1.parameters()},
                {'params': self.transformer_layer_norm.parameters()},
                {'params': self.final_fc1.parameters()},
                {'params': self.final_layer_norm.parameters()},
                {'params': self.final_output.parameters()},
            ],lr=self.lr_heads, amsgrad=True, weight_decay=0.01)
            
            scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=8)
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        },
                    }


    experiment_name=f'OneOutput-1024-CLIPv3-finalDropout-0.1'
    # Define the callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='../../../model_ckpts/Multimodal/CLIP+Longformer',
        filename=experiment_name,
        monitor='val_f1', mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping('val_f1', patience=20,mode='max')

    # instantiate the logger object
    logger = CSVLogger(save_dir="../../../logs/Multimodal/CLIP+Longformer", name=experiment_name)
    

    my_model=MultimodalClassifier()
    trainer=pl.Trainer(accelerator="gpu", devices=[1], deterministic=True, max_epochs=60, logger=logger, precision='16-mixed', accumulate_grad_batches=2,
                    callbacks=[lr_monitor, early_stopping, checkpoint_callback])
    trainer.fit(model=my_model,train_dataloaders=train_loader, val_dataloaders=validation_loader)

