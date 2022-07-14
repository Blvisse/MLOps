from collections import Counter
import ipywidgets as widgets
import itertools
import json
import pandas as pd
from urllib.request import urlopen
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from wordcloud import WordCloud, STOPWORDS
sns.set_theme()
warnings.filterwarnings("ignore")


#we load the data from @GoKuMohanda's github
try:
    url="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.json"
    print("Retrieving data from github...")
    url_payload=urlopen(url).read().decode('utf-8')

    # print(url_payload)
    projects=json.loads(url_payload)

    print("Data retrieved successfully!")
    
    print(f"Number of projects: {len(projects)}")
    print(json.dumps(projects[0],indent=2))
except Exception as e:
    print("Error: ",e)
    exit(1)

#from the data we get lets convert it to a DataFrame

data=pd.DataFrame(projects)
print(data.head())

#we retrieve the most commontags from the data
tags=Counter(data.tag.values)
print(tags.most_common())


#we load an auxiliary dataframe with the tags
try:
    url="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.json"
    tags_dict={}
    
    for item in json.loads(urlopen(url).read().decode('utf-8')):
        key=item.pop("tag")
        
        tags_dict[key]=item
        
    print(len(tags_dict))
    
    
except Exception as e:
    print("Error decoding tags: ",e)
    exit(1)
    
print(tags_dict['computer-vision'])

@widgets.interact(tag=list(tags_dict.keys()))
def display_tag_details(tag="computer-vision"):
    print (json.dumps(tags_dict[tag], indent=2))


#cary out some EDA on the data
#Check distribution of tags
tag, tags_count=zip(*Counter(data.tag.values).most_common())
plt.figure(figsize=(10,3))
ax=sns.barplot(list(tags), list(tags_count))
plt.title("Tag distribution", fontsize=20)
plt.xlabel("Tag", fontsize=16)
ax.set_xticklabels(tag,rotation=90, fontsize=14)
plt.ylabel("Number of Projects", fontsize=16)
plt.show()

#creating a wordcloud
@widgets.interact(tag=list(tag))
def display_wordcloud(tag="natural-language-processing"):
    
    plt.figure(figsize=(10,8))
    subset=data[data.tag==tag]
    text=subset.title.values
    wordcloud = WordCloud(stopwords=STOPWORDS,width=800, height=400, random_state=21, max_font_size=200).generate(" ".join(text))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
display_wordcloud()
