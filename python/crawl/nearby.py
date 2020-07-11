# target site : https://www.nearbuy.co.kr/

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from itertools import count
import pandas as pd
import re
from collections import OrderedDict

def product_list(code="여성의류", category="드레스"):
    base_url = "https://www.nearbuy.co.kr/"
    url = urljoin(base_url, "products_best")
    product_dict=OrderedDict()
    
    for page in count(1):
        params = {
            'sort':'NEW',
            'item_category':category.encode(),
            'shop_style_code':code.encode(),
            'page':page
        }
        print(f"I'm trying {page} page")
        
        html = requests.get(url, params=params).text
        soup = BeautifulSoup(html, 'html.parser')

        if soup.select(".row .empty_data"):
            print(f"Page maximum, {page-1} is the last page in this category")
            return product_dict

        for idx, tag in enumerate(soup.select(".row #productsUl a[href]")):            
            
            product_url = urljoin(base_url, tag['href'])
            # img_url
            img_tag = tag.find(class_='item_thumbnail')['style'];
            pattern = 'background: url\((https://.*)\) center'
            url_search = re.search(pattern, img_tag, re.IGNORECASE)
            if url_search: img_url = url_search.group(1)

            brand = tag.find(class_='store_name').text; brand=brand.split()[0]
            name = tag.find(class_='product_name').text;
            serial_number = tag['href'].split("/")[-1];
            price_lst = tag.find(class_='price_wrap').text.split(); price=" ".join(price_lst)

            #color&size
            prd_html = requests.get(product_url).text
            prd_soup = BeautifulSoup(prd_html, 'html.parser')
 
            tag_set = prd_soup.select('ul[class="essential dropeddown dnone"]')
            size_n_col = {value.select('li[data-option-name]')[0]['data-option-name']: " ".join(value.text.split()) for value in tag_set}
            print(size_n_col.keys())           
                        
            product = {
                'brand_name':brand,
                'product_name':name,
                'serial_number':serial_number,
                'price':price,
                'size and color':size_n_col,
                # 'color':color,
                'img_url':img_url,
                'product_url':product_url,
            }
            
            product_dict[product_url]=product
            print(product,"\n")

    return product_dict

skirt_dict = product_list(category="스커트") #142-143 pagination stop, skirt, women

There was a question of enviroment.

Since I implemented most of codes in Google Colab, you can show it using command `pip freeze`

