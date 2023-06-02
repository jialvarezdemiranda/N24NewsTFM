
import pandas as pd
import numpy as np
from transformers import AutoModel, CLIPProcessor, AutoConfig
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
from torch.utils.data import Dataset

SEED=1234542

pl.seed_everything(SEED, workers=True)

df_train=pd.read_csv('../../data/splitted/train.csv')
df_validation=pd.read_csv('../../data/splitted/validation.csv')
df_test=pd.read_csv('../../data/splitted/test.csv')

# Remove nan from caption column
df_train.fillna(value="", inplace=True)
df_validation.fillna(value="", inplace=True)
df_test.fillna(value="", inplace=True)


NUM_CLASSES= len(df_train['labels'].unique())

TEXT_USED='caption'

TRAIN_IMAGES_PATH= '../../images/train'
VALIDATION_IMAGES_PATH= '../../images/validation'
TEST_IMAGES_PATH= '../../images/test'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'openai/clip-vit-base-patch32'

pretrained_model = AutoModel.from_pretrained(MODEL_NAME)
clip_config= AutoConfig.from_pretrained(MODEL_NAME)
processor= CLIPProcessor.from_pretrained(MODEL_NAME)


class CustomMultimodalDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df= df
        self.img_dir = img_dir
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label_text= self.df['labels_text'].iloc[idx]
        img_path = os.path.join(self.img_dir, label_text, self.df['image_id'].iloc[idx])
        img_path=img_path + '.jpg'
        image = Image.open(img_path)
        if(image.mode != 'RGB'):
            image=image.convert('RGB')
        text = self.df[TEXT_USED].iloc[idx]
        label = self.df['labels'].iloc[idx]
        return image, text, label
    
train_dataset= CustomMultimodalDataset(df_train, TRAIN_IMAGES_PATH)
validation_dataset= CustomMultimodalDataset(df_validation, VALIDATION_IMAGES_PATH)
test_dataset= CustomMultimodalDataset(df_test, TEST_IMAGES_PATH)


class MultimodalCollator:
    HARD_IMG_AUGMENTER = T.RandAugment(num_ops=6, magnitude=9)
    SOFT_IMG_AUGMENTER = Compose([T.RandomPerspective(.1, p=.5),
                                  T.RandomHorizontalFlip(p=.5),
                                ])
    
    def __init__(self, processor=processor, augment_mode='hard', split='train', max_length=77):
        # 40 max length for vilt // 77 max length for clip
        self.processor = processor
        self.split = split
        self.max_length = max_length
        self.augment_mode = augment_mode

    def __call__(self, batch):
        images, texts, labels = list(zip(*batch))
        if self.split=='train' and self.augment_mode == 'hard':
            images = [self.HARD_IMG_AUGMENTER(img) for img in images]
        elif self.split=='train' and self.augment_mode == 'soft':
            images = [self.SOFT_IMG_AUGMENTER(img) for img in images]

        encoding = self.processor(images=images, 
                                  text=list(texts), 
                                  padding=True,
                                  max_length=self.max_length,
                                  truncation=True,
                                  return_tensors='pt')
        encoding['labels']=torch.tensor(labels)
        return encoding


BATCH_SIZE=8

collator_train=MultimodalCollator(split='train')
collator_val=MultimodalCollator(split='val')
collator_test=MultimodalCollator(split='test')
train_loader = DataLoader(train_dataset, collate_fn=collator_train, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, collate_fn=collator_val, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, collate_fn=collator_test, batch_size=BATCH_SIZE)



class CLIPClassifier(pl.LightningModule):
    def __init__(self, model=pretrained_model,  lr=2e-5):
        super(CLIPClassifier, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.lr=lr
        # En el train hacemos media de medias
        self.train_loss=[]
        self.train_accs=[]
        self.train_f1s=[]
        
        
        # Aqui computamos las métricas con todo para mayor precision   
        self.val_loss=[]             
        self.all_val_y_true=[]
        self.all_val_y_pred=[]
        
        self.model = model
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.fc1 = nn.Linear(512, 256)
        self.activation1 = nn.GELU()
        self.output = nn.Linear(256, NUM_CLASSES)
        
    def compute_outputs(self, input_ids, attention_mask, pixel_values):
        out_text=self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        out_image=self.model.get_image_features(pixel_values=pixel_values)
       
        combined_embed= abs(out_text-out_image) # Get the distance between the embeddings (absoluto de la resta)
        
        x = self.activation1(self.fc1(combined_embed))
        return self.output(x)
    
    def forward(self, batch):
        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        x = self.compute_outputs(input_ids, attention_mask, pixel_values)
        return x
    
    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        #Compute the output logits
        logits = self.compute_outputs(input_ids, attention_mask, pixel_values)
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
        labels = batch['labels']
        #Compute the output logits
        logits = self.compute_outputs(input_ids, attention_mask, pixel_values)
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
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=5)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    },
                }


experiment_name=f'CLIP_resta'
# Define the callbacks
checkpoint_callback = ModelCheckpoint(
     dirpath='../../model_ckpts/Multimodal/CLIPEmbeds',
     filename=experiment_name,
     monitor='val_f1', mode='max')
lr_monitor = LearningRateMonitor(logging_interval='epoch')
early_stopping = EarlyStopping('val_f1', patience=15,mode='max')

# instantiate the logger object
logger = CSVLogger(save_dir="../../logs/Multimodal/CLIPEmbeds", name=experiment_name)
 

my_model=CLIPClassifier(pretrained_model, lr=2e-3)
trainer=pl.Trainer(accelerator="gpu", devices=[1], deterministic=True, max_epochs=65, logger=logger, precision='16-mixed', accumulate_grad_batches=2,
                   callbacks=[lr_monitor, early_stopping, checkpoint_callback])
trainer.fit(model=my_model,train_dataloaders=train_loader, val_dataloaders=validation_loader)


