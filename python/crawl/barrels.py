# targetsite : http://www.barrels.co.kr

# 001000000000 - outer 
# 002000000000 - top
# 003000000000 - bottoms
# 004000000000 - shoes
# 005000000000 - hat
# 006000000000 - bag
# 007000000000 - accessories
# 008000000000 - life
# 012000000000 -  underwear


import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from itertools import count
import pandas as pd

#answer
from collections import OrderedDict

def product_list(category_id):
    base_url = "http://www.barrels.co.kr/categories/index/"
    url = urljoin(base_url, category_id)
    product_dict=OrderedDict()
    
    for page in count(1):
        params = {
            'page':page,
        }
        print(f"I'm trying {page} page")
        
        html = requests.get(url, params=params).text
        soup = BeautifulSoup(html, 'html.parser')
        
        if soup.select("#contents .no_content"):
            f"Page maximum, {page-1} is the last page in this category"
            return product_dict

        for tag in soup.select('#contents .item_listA a[href^=/product]'):            
            img_tag = tag.find('img')
            name_tag = img_tag['alt']    

            product_url = urljoin(url, tag['href'])
            try:
                brand, name = name_tag.split("]", maxsplit=1)
            except ValueError:
                brand=b=name=name_tag
                print(name_tag)
            img_url = img_tag['src']
            name, color = name.rsplit(" ", 1)

            b_tag = tag.select(".cont .price")[0]; price = b_tag.text
            c_tag = tag.select(".over .size")[0]; size = c_tag.text
            
            if product_url in product_dict:
                return product_dict
            
            product = {
                'brand_name':brand.strip('['),
                'product_name':name,
                'serial_number':(tag['href'].split('/'))[-1],
                'price':price,
                'size':size,
                'color':color,
                'img_url':urljoin(base_url, img_url),
                'product_url':product_url,
            }
            
            product_dict[product_url]=product
            # print(product)
    return product_dict

#change category if you want
outer_dict = product_list("001000000000") #outer category

outer_dict

len(outer_dict) #all outer amount

df = pd.DataFrame.from_dict(outer_dict, orient='index')
df = df.reset_index(drop=True); df.head(10)

df.columns

# df.to_csv("./test.csv", index=True)
df.to_csv("./test.csv", index=False)

pd.read_csv("./test.csv", ).head()

