import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime
import json


# get the labels into a separate column
raw_data = pd.read_json("final_data.json")
data = pd.DataFrame()
labels = []
for i in range(len((raw_data))):
    if raw_data["annotations"][i][0]["was_cancelled"]:
        labels.append([0])
    else:
        labels.append(raw_data["annotations"][i][0]["result"][0]["value"]["choices"])
data["labels"] = labels

# get the rest of the interesting metrics
metrics = ["created_at", "public_metrics.impression_count", "public_metrics.like_count", "author.username", "author.public_metrics.followers_count", "text_without_links", "photo_link"]
for metric in metrics:
    dictionary = []
    for i in range(len((raw_data))):
        dictionary.append(raw_data.data[i][metric])
    data[metric.rsplit('.')[-1]] = dictionary

# data.drop(["drafts", "predictions", "file_upload", "annotations", "data", "meta")

relevant = []

for i in range(len(data)):
    relevant.append(("Irrelevant" or 0) not in data["labels"][i] and data["username"][i] != ("United24media" and "tassagency_en"))

data["relevant"] = relevant

print(sum(data.relevant))

# ------- I HAVE A NICE DATAFRAME NOW :)

print(data.columns)

# # print(data.columns)
# authors = data["author.username"].tolist()

# unique, counts = np.unique(authors, return_counts=True)

# print(dict(zip(unique, counts)))

# # print(len(unique))


# data = data.sort_values(by=["created_at"], ascending=True)

# print(data.head(24)["author.username"])

# plt.figure(dpi=1200)

# fig, ax = plt.subplots()


# datelist = data["created_at"].tolist()
# converted_dates = list(map(datetime.datetime.strptime, datelist, len(datelist)*['%Y-%m-%dT%H:%M:%S.%fZ']))
# x_axis = converted_dates
# formatter = dates.DateFormatter('%Y-%m-%d')

# plt.hist(x_axis, bins=100, color="#1F6ABF")
# ax = plt.gcf().axes[0] 
# ax.xaxis.set_major_formatter(formatter)
# ax.set_ylabel("Number of tweets")
# ax.set_xlabel("Date")


# plt.gcf().autofmt_xdate(rotation=25)

# plt.savefig("graphs/dataset_hist.pdf")

# plt.show()