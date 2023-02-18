import pandas as pd
from glob import glob

## Cat
image_paths = glob("/mnt/d/project/pet_retina_develop/data/images/cat/*/*/*.jpg") + glob("/mnt/d/project/pet_retina_develop/data/images/cat/*/*/*.png")

labels = [img_path.split("/")[-3] + "_" +  img_path.split("/")[-2] for img_path in image_paths]

df = pd.DataFrame({
    "image_path" : image_paths,
    "label" : labels
})

df.to_csv("/mnt/d/project/pet_retina_develop/data/cat_full_dataframe.csv", index = False)

## dog
#image_paths = glob("/mnt/d/project/pet_retina/data/images/dog/*/*/*.jpg") + glob("/mnt/d/project/pet_retina/data/images/dog/*/*/*.png")

#labels = [img_path.split("/")[-3] + "_" +  img_path.split("/")[-2] for img_path in image_paths]

#df = pd.DataFrame({
#    "image_path" : image_paths,
#    "label" : labels
#})

#df.to_csv("/mnt/d/project/pet_retina/data/dog_full_dataframe.csv", index = False)
