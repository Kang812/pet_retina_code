import pandas as pd
from tqdm import tqdm

def class_aug_increase(df, labels, max_label_count, tpye):
    for i in tqdm(range(len(labels))):
        aug_img_paths = []
        aug_labels = []

        iter_num = 0

        sub_df = df[df['label'] == labels[i]]
        sub_label_counts = sub_df.shape[0]

        iteration = int(max_label_count/sub_label_counts)
        
        img_paths = sub_df['image_path'].to_list()
        labels_2 = sub_df['label'].to_list()

        for img_path, lb in zip(img_paths, labels_2):
            for i in range(iteration):
                if tpye == 'dog':
                    if iter_num % 2 == 0:
                        aug_img_paths.append(img_path)
                        aug_labels.append(lb)
                else:
                    aug_img_paths.append(img_path)
                    aug_labels.append(lb)
            iter_num += 1
        
        aug_df = pd.DataFrame({
            'image_path':aug_img_paths,
            'label' : aug_labels
        })

        aug_df['aug'] = "1"

        df = pd.concat([df, aug_df], axis=0)
    
    return df
