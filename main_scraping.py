"""
Scrape the desired websites and generate a list with news article and url 

"""



from help_functions import find_headings
import pandas as pd
import datetime
import openpyxl
import os
import pickle





folder_path = os.path.dirname(__file__)
print(folder_path)
dataframe_folder_path = os.path.join(folder_path,"news_dataframes")

news = []
data_dictionary = {}



## Create the txt file

filename = os.path.join(folder_path,"news",f"{datetime.datetime.today().strftime('%Y-%m-%d')}_filtered_news.txt")
with open(filename,"w") as f:
    f.write(filename)
    f.write("\n")

websites = ["https://www.hydrogenfuelnews.com",
            "https://h2lac.org/noticias",
            "https://www.globalhydrogenreview.com",
            "https://www.elpais.com.uy/sostenible",
            "https://bioenergytimes.com/",
            "https://www.hydrogeninsight.com/latest",
            "https://www.energiaestrategica.com/"]



if __name__ == "__main__":

    for website in websites:
        results = find_headings(website,filename,20)
        data_dictionary.update({website:results})
        for element in results:
            news.append(element)

    # news: List containing dicts of {"news" (article): ... , "url"}
    print(news)

    
    excel_path = os.path.join(folder_path,"news",f"{datetime.datetime.today().strftime('%Y-%m-%d')}_news.xlsx")
    dataf = pd.DataFrame(news)
    
    print(dataf)
    
    with open(os.path.join(dataframe_folder_path,f"{datetime.datetime.today().strftime('%Y-%m-%d')}_news.pkl"),'wb') as file:
        pickle.dump(dataf,file)


    dataf.to_excel(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active
    max_length = 0

    for cell in ws["B"]:
        width = len(str(cell.value)) 
        if width > max_length:
            max_length = width
    
    ws.column_dimensions["B"].width = max_length + 2
    wb.save(excel_path)


    

    
