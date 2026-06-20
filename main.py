
import time
import json
from src.homepage2vec.model import Webpage

from src.homepage2vec.model import WebsiteClassifier


model = WebsiteClassifier()


with open("input_all_html.json", 'r') as file:
    data = json.load(file)

    time_start_single = time.time()
    for item in data:
        website = Webpage(item["url"])
        website.html = item["html"]

        if website.html != None:
            try:
                scores, embeddings = model.predict(website)
            except:
                print("Error with " + item["url"])
    time_end_single = time.time()

print("Delta single: " + str(time_end_single-time_start_single))

