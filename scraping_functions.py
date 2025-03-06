"""
Scraping functions
"""
import os
import bs4
import os
import requests
import pandas as pd
import datetime
import os


from helper_functions import save_df_as_xlsx




def check_complete_url(website_url, link):
    """
    Checks if the link is complete or relative

    Parameters:
    website_url (str): The website where the link was found
    link (str): The link to access
    
    Returns:
    full_url (str): The full url of the link 
    """
    if link is not None:
        full_url = ""
        if link[:5] != "https":
            full_url = website_url + link
        else:
            full_url = link
        return full_url
    else:
        return " "

def contains_keyword(text, keywords):
    """
    Checks if the text contains at least one of the keywords. If there are no keywords, it returns True
    Args:
    Text (str): Text to be checked for the keywords
    keywords(list): List of keyword strings
    Returns:
    True if at least one keyword in text, else False
    """
    if keywords is None:
        return True
    else:
        text = text.lower()
        for keyword in keywords:
            keyword = keyword.lower()
            if str(keyword + " ") in text or str(" " + keyword) in text:
                return True
        return False

def find_news_article_titles(website_url,txt_write,max_amount_of_news,keywords):

    """
    Scrapes the desired website url and filters the news according to the keywords.
    
    Args:
    website_url (str): The website to be scraped
    website_txt (str): FOR TESTING
    keywords (list): List of keywords to filter 
    txt_write (str): Txt file to write the filtered news
    max_amount_of_news (int): Maximum number of news to filter from the given website
    headers (bool): If True, we search the headers. Otherwise, we search for all hyperlinks

    Returns: List of dictionaries with the news headlines and urls
    
    
    """
    
    news = []

    
    #To be tested locally at home
    try:
        req = requests.get(website_url)
        website = req.text
    except:
        print("Could not access the website")
        return news
            
    soup = bs4.BeautifulSoup(website)

    
    tags = soup.find_all(["h2", "h3"])
    div_header = soup.find("div", class_="widget-title")
    if div_header:
        div_header.decompose()
    i = 0  # Counter to stop printing more than max_amount_of_news
    for link in tags:
        url = None
        if i < max_amount_of_news:
            text = link.get_text()
            if contains_keyword(text, keywords): # If keywords are not specified, it returns True

                if i == 0: # Write the url of the website only once
                    with open(txt_write, "a", encoding="utf-8") as f:
                        f.write(website_url)
                        f.write("\n")
                
                
                with open(txt_write, "a", encoding="utf-8") as f: # Write the headline
                    f.write(text.strip())
                    f.write("\n")
                    h = link.find("a")
                    if h is not None:
                        url = check_complete_url(website_url,h["href"]) 
                        f.write(url)
                    f.write("\n")

                news.append({"text":text.strip(),
                                    "url": url})
    
            i += 1
    if i != 0: # Separate the different websites
        with open(txt_write, "a", encoding="utf-8") as f:
            f.write("\n")
    
    return news


def scrape_and_save(websites,news_folder_path, keywords=None, max_amount_of_news=20):
    
    """
    Scrape the provided websites for news articles and save the results as a txt and as an Excel file.

    Args:
        websites (list): A list of website URLs to scrape for news articles.
        news_folder_path (str): The path to the folder where the news articles will be saved.
        keywords (list): A list of keywords to filter the news articles.
    Returns:
        news_dataframe (pd.DataFrame): A DataFrame containing the scraped news articles.

    The function performs the following steps:
    1. Initializes an empty list to store news articles and a dictionary to store results by website.
    2. Creates a text file to log the news articles.
    3. Iterates over each website, scrapes the news articles, and updates the dictionary and list.
    4. Saves the news articles to a pickle file.
    5. Saves the news articles to an Excel file.
    6. Adjusts the column width in the Excel file for better readability.
    """
    
    
    news = []
    

    ## Create the txt file
    filename = os.path.join(news_folder_path,f"{datetime.datetime.today().strftime('%Y-%m-%d')}_news.txt")
    with open(filename,"w") as f:
        f.write(filename)
        f.write("\n")

    for website in websites:
        results = find_news_article_titles(website,filename,max_amount_of_news,keywords)
        for element in results:
            news.append(element)


    # save an Excel file with the unfiltered news
    excel_path = os.path.join(news_folder_path,f"{datetime.datetime.today().strftime('%Y-%m-%d')}_news.xlsx")
    news_dataframe = pd.DataFrame(news)     

    save_df_as_xlsx(news_dataframe,excel_path)
    
    return news_dataframe

