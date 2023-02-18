import json
import pandas as pd

from glob import glob
from tqdm import tqdm

def json_to_dataframe(image_paths, output_path):
    image_path_list = []
    label_list = []

    for i in tqdm(range(len(image_paths))):
        label_disease_list = []
        json_path = image_paths[i].replace(".jpg", ".json").replace(".png",".json")
        image_path_list.append(image_paths[i])
        kind = json_path.split("/")[-4]
        label_disease = json_path.split("/")[-3]
        um = json_path.split("/")[-2]

        with open(json_path, "r") as f:
            annotations = json.load(f)
        
        label = kind + "_" + annotations['images']['meta']['eye_position'] + "_" + label_disease + "_"+ um
        
        label_list.append(label)

    df = pd.DataFrame({
        "image_path" : image_path_list,
        "label" : label_list
    })
    
    df.to_csv(output_path, index=False)


print("Train DataFrame Make:")
image_paths = glob("/mnt/d/project/pet_retina/data/train/*/*/*/*.jpg") + glob("/mnt/d/project/pet_retina/data/train/*/*/*/*.png")
output_path = "/mnt/d/project/pet_retina/data/full_data_frame.csv"
json_to_dataframe(image_paths, output_path)
