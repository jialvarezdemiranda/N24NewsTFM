
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

if __name__== '__main__':

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
        # HARD_IMG_AUGMENTER = T.RandAugment(num_ops=6, magnitude=9)
        # SOFT_IMG_AUGMENTER = Compose([T.RandomPerspective(.1, p=.5),
        #                               T.RandomHorizontalFlip(p=.5),
        #                             ])
        
        def __init__(self, processor=processor, augment_mode='hard', split='train', max_length=77):
            # 40 max length for vilt // 77 max length for clip
            self.processor = processor
            self.split = split
            self.max_length = max_length
            self.augment_mode = augment_mode

        def __call__(self, batch):
            images, texts, labels = list(zip(*batch))
            # if self.split=='train' and self.augment_mode == 'hard':
            #     images = [self.HARD_IMG_AUGMENTER(img) for img in images]
            # elif self.split=='train' and self.augment_mode == 'soft':
            #     images = [self.SOFT_IMG_AUGMENTER(img) for img in images]

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
    train_loader = DataLoader(train_dataset, collate_fn=collator_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, prefetch_factor=8, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, collate_fn=collator_val, batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, collate_fn=collator_test, batch_size=BATCH_SIZE,num_workers=4, prefetch_factor=8, pin_memory=True)

    for batch in train_loader:
        print(batch)
        break



