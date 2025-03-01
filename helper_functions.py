import openpyxl



def save_df_as_xlsx(dataframe,path):
    """
    Saves a pandas DataFrame to an Excel file and adjusts the column width for better readability.
    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        path (str): The file path to save the DataFrame.
    """


    dataframe.to_excel(path)

    wb = openpyxl.load_workbook(path)
    ws = wb.active
    max_length = 0


    #Adjust column length
    for cell in ws["B"]:
        width = len(str(cell.value)) 
        if width > max_length:
            max_length = width
    
    ws.column_dimensions["B"].width = max_length + 2
    wb.save(path)
    wb.close()