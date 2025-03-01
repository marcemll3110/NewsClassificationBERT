from bert_ordinal_regression import BERTModel
import torch
import os
import pickle
import numpy as np
import openpyxl
import datetime

from coral_pytorch.dataset import corn_label_from_logits



if __name__ == "__main__":
    root = os.getcwd()
    print(root)
    dataset_path = os.path.join(root, "dataset","dataset_complete_class.pkl")
    unfiltered_pickle_path = os.path.join(root, "news_dataframes", "2025-02-11_news.pkl")
    filtered_excel_path = os.path.join(root, "news", "2025-02-21_filtered_news.xlsx")
    excel_path = os.path.join(root, "news", f"{datetime.datetime.today().strftime('%Y-%m-%d')}_news.xlsx")
    model_path = os.path.join(root, "models", "2025-02-22_bert_state_dict_CORN_2.pth")
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    b = BERTModel("Robertito", device)
    b.build_model(num_classes=4)

    b.load_model(model_path)
    b.model.eval()

    with open(unfiltered_pickle_path,"rb") as test_file:
        pred_text = pickle.load(test_file)
    
    


    
    #b.train_model(dataset_path, num_epochs=100,batch_size=16, lr=1e-4)
    # dataset_path,epochs,batch_size,DEVICE,NUM_CLASSES
    

    
    #b.save_model(model_path)

    pred_text = "Carbon capture project initial design"
    predictions = b.predict_message(pred_text)

    

    with torch.no_grad():
        probas = torch.sigmoid(predictions)
        probas = torch.cumprod(probas, dim=1)
        print(probas)
    predicted_labels = corn_label_from_logits(predictions).float()
    print(predicted_labels)

    """
   
    #print(predictions)
    pred_text["scores"] = predicted_labels
    #pred_text = np.round(predictions,2)
    #pred_text.to_excel(excel_path)

    print(pred_text[pred_text["scores"]==0])
    
    
    pred_text_filtered = pred_text

    pred_text_filtered.to_excel(filtered_excel_path)

    wb = openpyxl.load_workbook(filtered_excel_path)
    ws = wb.active
    max_length = 0

    for cell in ws["B"]:
        width = len(str(cell.value)) 
        if width > max_length:
            max_length = width
    
    ws.column_dimensions["B"].width = max_length + 2
    wb.save(filtered_excel_path)
    """

