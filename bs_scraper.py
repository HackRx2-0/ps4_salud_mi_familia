from bs4 import BeautifulSoup
import requests
import re
import pandas as pd


def save_soup(save_str):
    with open('test.txt', 'w') as f:
        f.write(save_str.encode('utf-8').decode('ascii', 'ignore'))


# ------------------------- Scrape Text from Links df ------------------------ #
links_df = pd.read_csv('sublinks.csv')
links_list = links_df['link'].tolist()

# def remove_unwanted():
#     for i in soup.find_all("span"):
#         if 'class' in i.attrs:
#             if "unwanted" in i.attrs['class']:
#                 print(i.text)


def add_text(row):
    if row['link'][0] == '/':
        href = row['link']
        print(f'https://www.bajajfinserv.in{href}')
        html = requests.get(f'https://www.bajajfinserv.in{href}')
        soup = BeautifulSoup(html.text, 'lxml')

        p_tag = soup.find_all('p')
        p_text = []
        exclude_tags = ['hdrsearchResultGotoTag', 'v1_langformobile']

        for p in p_tag:
            if 'class' in p.attrs:
                if any(tag in p.attrs['class'] for tag in exclude_tags):
                    continue

            else:
                p_text.append(p.text)

        # p_text = [p.text for p in p_tag]
        return p_text
    else:
        return 'NA'


def clean_text(text):
    exclude_text = ['Account Details']
    text = [x.replace('\r\n\t\t\t\t\t', '') for x in text]
    text = [x.replace('\n', ' ') for x in text]
    for i in exclude_text:
        text = [x.replace(i, '') for x in text]
    text = ' '.join(text)
    return text


links_df['content'] = links_df.apply(add_text, axis=1)
links_df['content'] = links_df['content'].apply(clean_text)

links_df.to_csv('sublinks_df_content.csv', index=False)

# ---------------------------------------------------------------------------- #


# html = requests.get("https://www.bajajfinserv.in/emi-network-health-emi-card")
# soup = BeautifulSoup(html.text, "lxml")

# # atag=soup.find_all('a', class_=re.compile("-ab"))
# ptag = soup.find_all('p')

# v1_items = soup.select("div[class*='v1_subSub']")

# for p in ptag:
#     print(p.text)

# itemtitle = []
# itemprice = []
# for a in atag:
#   for title,price in zip(a.find_all('div', class_=re.compile("-m")),a.find_all('div', class_=re.compile("-k"))):
#       itemtitle.append(title.text)
#       itemprice.append(price.find('div').text)

# df=pd.DataFrame({"Title" :itemtitle, "Price" : itemprice})
# print(df)

# for v1 in v1_items:
#     if len(v1.text) > 100:
#         print(v1.text.replace(' ', ''))
#         print(type(v1.text))
# input('')
# print(soup.prettify())

# print(soup.get_text())
# print(BeautifulSoup(v1_items, "lxml").get_text())

# save_soup(p_text_list)
