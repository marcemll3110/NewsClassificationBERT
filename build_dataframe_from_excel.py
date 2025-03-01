"""
The dataset for training the model is an excel spreadsheet. The author of this code graded the news articles based on his personal interests.
These functions serve to read the excel spreadsheet, tokenize it and save them as as dict it in a pickle file.
"""



import pandas as pd
import os
from transformers import DistilBertTokenizerFast
import pickle




root = os.getcwd()
print(root)
excel_path = os.path.join(root,"dataset" ,"NewsCompDataset.xlsx")
save_path = os.path.join(root, "dataset","dataset_complete.pkl")
print(save_path)
print(excel_path)



def excel_to_df(path,ordinal):
    """
        Reads an Excel file and returns specific columns as pandas DataFrame.
        Parameters:
        path (str): The file path to the Excel file.
        ordinal (bool): A flag to determine whether to return the 'Class' column.
        Returns:
        tuple: A tuple containing the 'text' column (with NaN values filled with a space) 
               and the 'Interesting' column. If ordinal is True, also includes the 'Class' column.
    """

    pd_df = pd.read_excel(path)
    print(pd_df)
    if ordinal:
        return pd_df["text"].fillna(" "), pd_df["Interesting"], pd_df["Class"]
    else:
        return pd_df["text"].fillna(" "), pd_df["Interesting"]




def tokenize(path,ordinal):
    """
    Tokenizes the titles from an Excel file using the DistilBertTokenizerFast.
    Args:
        path (str): The file path to the Excel file containing the data.
        ordinal (bool): A flag to determine whether to return the 'Class' column.
    Returns:
        dict: A dictionary containing the following keys:
            - "encodings" (BatchEncoding): The tokenized titles.
            - "labels" (list): The list of labels corresponding to the titles.
            - "class" (list): The list of classes corresponding to the titles.
    """
    

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    if ordinal:
        titles, labels, cl = excel_to_df(path,ordinal=ordinal)
        cl = list(cl)
    else:
        titles, labels = excel_to_df(path,ordinal=False)
        titles = list(titles)
        labels = list(labels)
    

    encodings = tokenizer(titles, truncation=True, padding=True)
    #dataset_final = hfDataset(encodings, labels)
    #encodings = np.array(encodings)
    #labels = np.array(labels)
    if ordinal:
        return {"encodings": encodings, "labels": labels, "class": cl}
    else:
        return {"encodings": encodings, "labels": labels}




def save_dataset(excel_path, save_path):
    with open(save_path, "wb") as file:
        pickle.dump(tokenize(excel_path), file)











