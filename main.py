
import time
import json
from src.homepage2vec.model import Webpage

from src.homepage2vec.model import WebsiteClassifier


model = WebsiteClassifier()


with open("input_all_html.json", 'r') as file:
    data = json.load(file)

websites = []
for item in data:
    if item["html"] != None:
        website = Webpage(item["url"])
        website.html = item["html"]
        websites.append(website)

#time_start_single = time.time()
#for website in websites:
#    print("Start "+website.url)
#    try:
#        scores, embeddings = model.predict(website)
#    except Exception as e:
#        print("Error with " + website.url+": "+str(e))
#time_end_single = time.time()
#print("Delta single: " + str(time_end_single-time_start_single))

time_start_batch = time.time()
scores = model.predict_batch(websites)
time_end_batch = time.time()
print("Delta batch: " + str(time_end_batch-time_start_batch))
print(scores)


