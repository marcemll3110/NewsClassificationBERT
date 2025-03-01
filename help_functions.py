"""
help functions
"""
import os
import bs4
import os
import requests


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
    Checks if the text contains at least one of the keywords
    Args:
    Text (str): Text to be checked for the keywords
    keywords(list): List of keyword strings
    Returns:
    True if at least one keyword in text, else False
    """
    text = text.lower()
    for keyword in keywords:
        keyword = keyword.lower()
        if str(keyword + " ") in text or str(" " + keyword) in text:
            return True
    return False

def find_headings(website_url,txt_write,max_amount_of_news):

    """
    Scrapes the desired website url and filters the news according to the keywords.
    
    Args:
    website_url (str): The website to be scraped
    website_txt (str): FOR TESTING
    keywords (list): List of keywords to filter 
    txt_write (str): Txt file to write the filtered news
    max_amount_of_news (int): Maximum number of news to filter from the given website
    headers (bool): If True, we search the headers. Otherwise, we search for all hyperlinks

    Returns: Dict containing "news" and "url"
    
    
    """
    
    news = []

    
    #To be tested locally at home
    req = requests.get(website_url)
    website = req.text
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
            #if contains_keyword(text, keywords):
            #print(link)
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

