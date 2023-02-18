import pandas as pd
from sklearn.model_selection import train_test_split


disease = ["결막염_무", "결막염_유"]

data = pd.read_csv("/mnt/d/project/pet_retina_develop/data/cat_full_dataframe.csv")

sample_data = data[(data['label'] == disease[0]) | (data['label'] == disease[1])]
sample_data = sample_data.reset_index(drop=True)

train, val = train_test_split(sample_data, test_size=0.2, stratify = sample_data['label'], random_state=2022)
val, test = train_test_split(val, test_size=0.5, stratify = val['label'], random_state=2022)

train.to_csv("/mnt/d/project/pet_retina_develop/data/cat_train_conjunctivitis.csv", index = False)
val.to_csv("/mnt/d/project/pet_retina_develop/data/cat_val_conjunctivitis.csv", index = False)
test.to_csv("/mnt/d/project/pet_retina_develop/data/cat_test_conjunctivitis.csv", index = False)