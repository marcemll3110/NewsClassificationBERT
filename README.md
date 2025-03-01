# NewsClassificationBERT

# Scraper + DISTILBERT for Linear Regression

## Scraper

This Python project contains a simple scraper using `requests` and `beautifulsoup4`. The scraper can enter the desired websites and write the titles of the news and the corresponding URLs in an Excel file. It is also possible to filter websites using keywords.

## DISTILBERT with Regression Head

After several weeks of gathering the news, **690 titles** were collected and graded by the creator of the script. The goal was to see if the model could learn the preferences of the user. The gradings are a numerical value from **0 to 1**, with:

- `0`: not interesting at all
- `1`: extremely interesting

### Approaches Tested

Two different approaches were tested:

1. **Linear Regression Head**  
   A linear regression head was added to the DISTILBERT model. **Mean Squared Error (MSE)** was selected as the loss function, and the model was trained.
   
2. **Ordinal Regression**  
   Four classes were created based on the gradings of the news articles:
   - `0`: not interesting at all
   - `4`: extremely interesting  
   
   For ordinal regression, the loss was based on the findings in the paper:  
   *Deep Neural Networks for Rank-Consistent Ordinal Regression Based on Conditional Probabilities* – **Shi et al.**

---

It is hard to classify and explain what makes a news article "interesting" versus "very interesting." However, in the case of regression, the model seems to learn that certain words should increase the rating of an article. 

For example, the creator is currently based in **Uruguay**. The neural network recognized that articles with the keyword **"Uruguay"** should have a higher score than articles mentioning other countries. 

A simple test was carried out:
- An invented title, not seen by the model, was graded twice:
  1. Ending with "in Uruguay"
  2. Ending with "in India"  

The first example consistently outscored the second one in all tests.

---

## To-Do

- More thorough and systematic testing of the model is required.
- The scraper sometimes crashes when a website makes too many redirects. A limit on the number of redirects needs to be implemented.

## Dependencies

- `beautifulsoup4`
- `requests`
- `pandas`
- `openpyxl`
- `pytorch`
- `transformers`
- `pickle`

