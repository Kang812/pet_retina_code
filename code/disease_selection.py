import pandas as pd
from sklearn.model_selection import train_test_split


disease = ["비궤양성각막염_무", "비궤양성각막염_유"]

data = pd.read_csv("/mnt/d/project/pet_retina_develop/data/cat_full_dataframe.csv")

sample_data = data[(data['label'] == disease[0]) | (data['label'] == disease[1])]
sample_data = sample_data.reset_index(drop=True)

train, val = train_test_split(sample_data, test_size=0.2, stratify = sample_data['label'], random_state=2022)
val, test = train_test_split(val, test_size=0.5, stratify = val['label'], random_state=2022)

#train.to_csv("/mnt/d/project/pet_retina_develop/data/cat_train_meibomian_gland.csv", index = False)
#val.to_csv("/mnt/d/project/pet_retina_develop/data/cat_val_meibomian_gland.csv", index = False)
#test.to_csv("/mnt/d/project/pet_retina_develop/data/cat_test_meibomian_gland.csv", index = False)

train.to_csv("/mnt/d/project/pet_retina_develop/data/cat_train_non_ulcerative_corn_disease.csv", index = False)
val.to_csv("/mnt/d/project/pet_retina_develop/data/cat_val_non_ulcerative_corn_disease.csv", index = False)
test.to_csv("/mnt/d/project/pet_retina_develop/data/cat_test_non_ulcerative_corn_disease.csv", index = False)