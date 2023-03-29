import pandas as pd
from sklearn.model_selection import train_test_split

## Cat
full_df = pd.read_csv("/mnt/d/project/pet_retina/data/cat_full_dataframe.csv")

train, val = train_test_split(full_df, test_size=0.2, stratify=full_df['label'], random_state=11)
val, test = train_test_split(val, test_size=0.5, stratify=val['label'], random_state=0)

print("train shape:",train.shape)
print("valid shape:",val.shape)
print("test shape:",test.shape)

train.to_csv("/mnt/d/project/pet_retina/data/cat_train.csv", index = False)
val.to_csv("/mnt/d/project/pet_retina/data/cat_val.csv", index = False)
test.to_csv("/mnt/d/project/pet_retina/data/cat_test.csv", index = False)
print("Cat Split Complete !")
print()

## Dog
full_df = pd.read_csv("/mnt/d/project/pet_retina/data/dog_full_dataframe.csv")

train, val = train_test_split(full_df, test_size=0.2, stratify=full_df['label'], random_state=11)
val, test = train_test_split(val, test_size=0.5, stratify=val['label'], random_state=0)

print("train shape:",train.shape)
print("valid shape:",val.shape)
print("test shape:",test.shape)

train.to_csv("/mnt/d/project/pet_retina/data/dog_train.csv", index = False)
val.to_csv("/mnt/d/project/pet_retina/data/dog_val.csv", index = False)
test.to_csv("/mnt/d/project/pet_retina/data/dog_test.csv", index = False)
print("Dog Split Complete !")

