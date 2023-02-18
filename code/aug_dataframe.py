import sys
import pandas as pd
sys.path.append("/mnt/d/project/pet_retina/utils")
from aug_df_make import class_aug_increase

## Cat
print("Cat:")
labels = ['비궤양성각막염_무', '비궤양성각막염_유', '안검염_무', '안검염_유']
max_label_count = 2500

train_df = pd.read_csv("/mnt/d/project/pet_retina/data/cat_train.csv")
train_df['aug'] = "0"

new_train_df = class_aug_increase(train_df, labels, max_label_count, 'cat')
new_train_df.to_csv("/mnt/d/project/pet_retina/data/cat_train_aug.csv", index = False)

## Dog
print("Dog:")
labels = ["색소침착성각막염_유", "색소침착성각막염_무", "백내장_비성숙", "궤양성각막질환_하", "안검염_유",
          "궤양성각막질환_무","백내장_무","안검염_무","백내장_성숙","백내장_초기","궤양성각막질환_상","비궤양성각막질환_하",
          "안검종양_유","비궤양성각막질환_무","안검종양_무","비궤양성각막질환_상"]

max_label_count = 7600
train_df = pd.read_csv("/mnt/d/project/pet_retina/data/dog_train.csv")
train_df['aug'] = "0"
new_train_df = class_aug_increase(train_df, labels, max_label_count, 'dog')

labels = ["안검종양_유","비궤양성각막질환_무","안검종양_무","비궤양성각막질환_상", "비궤양성각막질환_하"]
new_train_df = class_aug_increase(new_train_df, labels, max_label_count, tpye = 'dog')
new_train_df.to_csv("/mnt/d/project/pet_retina/data/dog_train_aug.csv", index = False)


