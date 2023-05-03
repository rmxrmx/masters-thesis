import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime

data = pd.read_csv("csvs/formatted_data.csv")
print(data["author.username"].unique(), data["author_id"].unique().tolist())




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