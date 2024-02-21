import requests
import numpy as np
import pandas as pd
import os
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
WEBPAGE = requests.get('https://en.wikipedia.org/wiki/List_of_Canadian_provinces_and_territories_by_historical_population')

def get_data(page):
    soup = BeautifulSoup(page.text, 'html.parser')
    tables = soup.find_all('table', class_='wikitable sortable')
    data = []
    hyperlinks = []

    # Get Table Data:
    for table in tables:
        header_data = []
        row_data = []
        
        header_row = table.find('tr') # get the header row
        headers = header_row.find_all('th') # get the headers
        for header in headers:
            header_data.append(header.text.rstrip())
            
        rows = table.find_all('tr')[1:] # get the rows
        for row in rows:
        
            # Get HyperLinks:
            row_cells = row.find_all('td')
            for cell in row_cells:
                links = cell.find_all('a') # get all hyperlinks from tables
                for link in links:
                    href = link.get('href')
                    if (not href.startswith('#cite_note')): 
                        hyperlinks.append(href)

            row_values = [row_cell.text.rstrip() for row_cell in row_cells]
            row_data.append(row_values)
        frame = pd.DataFrame(row_data, columns=header_data)
        frame = frame.replace(r'^\s*$', np.nan, regex=True)
        
        for col in frame.columns:
            if col == "Name":
                frame[col] = frame[col].str.replace(r"\[.*\]", "", regex=True)
                frame[col] = frame[col].str.replace(r"\xa0", "", regex=True) #remove the non-breaking spaces
                
            else:
                frame[col] = frame[col].str.replace(r"\[.*\]", "",regex=True) #remove the references
                frame[col] = frame[col].str.replace(r",", "", regex=True) #remove the commas
                frame[col] = frame[col].str.replace(r"\s", "", regex=True) #remove the spaces
                frame[col] = frame[col].str.replace(r"\xa0", "", regex=True) #remove the non-breaking spaces
                frame[col] = frame[col].str.replace(r"\.", "",regex=True) #remove the periods
                frame[col] = frame[col].str.replace(r"\*", "", regex=True)  #remove the asterisks
                frame[col] = frame[col].str.replace(r'\[.*\]', '',regex=True) #remove the references
        data.append(frame)

    combined_frame = pd.concat([cur_frame.set_index("Name") for cur_frame in data],axis=1) #concatonate the dataframes based off the name column
    combined_frame = combined_frame.reset_index() #reset the index
    combined_frame = combined_frame[combined_frame["Name"] != "Canada" ] #remove the canada and total row
    combined_frame = combined_frame[combined_frame["Name"] != "Total" ] #remove the canada and total row
    combined_frame.rename(columns={'Confederated[d]': 'Confederated'}, inplace=True) #rename the column to remove the references
    return combined_frame, hyperlinks

def download_links(links: list):
    for link in links:
        url = f'https://en.wikipedia.org{link}'
        response = requests.get(url)
        if response.status_code == 200:
            filename = os.path.join("pages", link.split('/')[-1] + '.html')
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {url}")
        else:
            print(f"Failed to download: {url}")

    return

data, hyperlinks = get_data(WEBPAGE)
download_links(hyperlinks)


