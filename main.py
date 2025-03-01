from bert_regression import BERTModel
import torch
import os
import pickle
import numpy as np
import openpyxl
import datetime




if __name__ == "__main__":
    root = os.getcwd()
    print(root)
    dataset_path = os.path.join(root, "dataset","dataset_complete.pkl")
    unfiltered_pickle_path = os.path.join(root, "news_dataframes", "2025-02-11_news.pkl")
    filtered_excel_path = os.path.join(root, "news", "2025-02-11_filtered_news.xlsx")
    excel_path = os.path.join(root, "news", f"{datetime.datetime.today().strftime('%Y-%m-%d')}_news.xlsx")
    model_path = os.path.join(root, "models")
    checkpoint_path = os.path.join(root, "models", "best_model.pt")

    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    b = BERTModel("Robertito", device)
    #b.build_model()
    #b.load_model(model_path)
    


    
    #b.train_regression_model(dataset_path,checkpoint_dir=model_path,num_epochs=25)
    b.load_checkpoint(checkpoint_path)
    #b.model.eval()



    with open(dataset_path,"rb") as test_file:
        pred_text = pickle.load(test_file)
    


    pred_text = "Green hydrogen project in Uruguay has reached FID"
    predictions = b.predict_message(pred_text)
    print(predictions)
    predictions = predictions.detach().cpu().numpy()
 


