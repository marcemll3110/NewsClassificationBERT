import pandas as pd
import openpyxl
import os
from transformers import DistilBertTokenizerFast
import torch
import numpy as np
import pickle


root = os.getcwd()
print(root)
excel_path = os.path.join(root,"dataset" ,"NewsCompDataset_class.xlsx")
save_path = os.path.join(root, "dataset","dataset_complete_class.pkl")
print(save_path)
print(excel_path)



def excel_to_df(path):

    pd_df = pd.read_excel(path)
    print(pd_df)
    return pd_df["text"].fillna(" "), pd_df["Interesting"], pd_df["Class"]




def tokenize(path):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    titles, labels, cl = excel_to_df(path)
    titles = list(titles)
    labels = list(labels)
    cl = list(cl)

    encodings = tokenizer(titles, truncation=True, padding=True)
    #dataset_final = hfDataset(encodings, labels)
    #encodings = np.array(encodings)
    #labels = np.array(labels)
    return {"encodings": encodings, "labels": labels, "class": cl}




def save_dataset(excel_path, save_path):
    with open(save_path, "wb") as file:
        pickle.dump(tokenize(excel_path), file)


save_dataset(excel_path, save_path)









