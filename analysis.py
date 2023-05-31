#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime
import requests
import json
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.stats import entropy, shapiro, spearmanr, pearsonr
import textdescriptives as td
import seaborn as sns
import pingouin as pg
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.feature_selection import f_regression, f_classif
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret.blackbox import ShapKernel
from interpret import show
from PIL import Image
import matplotlib.patches as mpatches
import ptitprince as pt

pd.set_option('display.max_colwidth', None)

data2 = pd.read_csv("csvs/formatted_data.csv")
authors = data2["author.username"].tolist()

unique, counts = np.unique(authors, return_counts=True)

# print(dict(zip(unique, counts)))
# get the labels into a separate column

raw_data = pd.read_json("final_data.json")
data = pd.DataFrame()
labels = []
remove = []
for i in range(len((raw_data))):
    if raw_data["annotations"][i][0]["was_cancelled"]:
        labels.append([0])
        remove.append(True)
    else:
        labels.append(raw_data["annotations"][i][0]["result"][0]["value"]["choices"])
        remove.append(False)
data["labels"] = labels
data["remove"] = remove
# get the rest of the interesting metrics
metrics = ["created_at", "public_metrics.impression_count", "public_metrics.like_count", "author.username", "author.public_metrics.followers_count", "text_without_links", "photo_link"]
for metric in metrics:
    dictionary = []
    for i in range(len((raw_data))):
        dictionary.append(raw_data.data[i][metric])
    data[metric.rsplit('.')[-1]] = dictionary

data = data[data["remove"] != True]
data.drop(columns=["remove"], inplace=True)
data.reset_index(drop=True, inplace=True)

relevant = []
for i in range(len(data)):
    relevant.append(("Irrelevant" not in data["labels"][i] and ["Text"] != data["labels"][i] and ["Portrait"] != data["labels"][i] and ["Bandwagon"] != data["labels"][i]))

data["relevant"] = relevant
data = data[data["username"] != "United24media"]
data = data[data["username"] != "tassagency_en"]

# TODO: remove [0] from dataset
# drop the only datapoint in early war
data.drop(1851, inplace=True)

data["photo_paths"] = data["photo_link"].apply(lambda x: x.split('/')[-1])


data.sort_values("created_at", inplace=True)
data["labels"] = data["labels"].apply(lambda j: list(filter(lambda x: x != "Bandwagon", j)))

data_with_irrelevant = data.copy()
data = data[data["relevant"] == True]
data = data.reset_index(drop=True)



# ------- I HAVE A NICE DATAFRAME NOW :)
data.to_csv("csvs/final_dataset.csv")



# reformatting the data for the experimental setup

data["labels_no_mode"] = data["labels"].apply(lambda j: list(filter(lambda x: x not in ("Text", "Portrait"), j)))

data_modelling = data[["labels_no_mode", "text_without_links", "photo_paths"]]

data_modelling = data_modelling.rename(columns={"labels_no_mode" : "labels", "text_without_links" : "text", "photo_paths" : "image", "index": "id"})
data_modelling.to_csv("csvs/modelling_data.csv", index=False)




authors = data["username"].tolist()

unique, counts = np.unique(authors, return_counts=True)

# print(dict(zip(unique, counts)))
# print(len(unique))


occurences = {}
for i in range(len(data)):
    number = len(data.iloc[i]["labels_no_mode"])
    if number in occurences:
        occurences[number] += 1
    else:
        occurences[number] = 1

# print(occurences)

# get all the unique labels
label_types_all = []
for i in range(len(data)):
    for j in data.iloc[i]["labels_no_mode"]:
        label_types_all.append(j)

label_types = np.unique(label_types_all)

label_types_all = []
for i in range(len(data)):
    for j in data.iloc[i]["labels"]:
        label_types_all.append(j)

# print(Counter(label_types_all))
# my_probabilities = [x / len(label_types_all) for x in Counter(label_types_all).values()]
# print(entropy(my_probabilities))

# dimitrov_labels = [492, 347, 602, 111, 100, 70, 91, 67, 112, 55, 14, 36, 27, 26, 40, 35, 23, 7, 7, 5, 95, 90]
# dimitrov_probabilities = [x / sum(dimitrov_labels) for x in dimitrov_labels]
# print(entropy(dimitrov_probabilities))

# binary labels for each row
for label in label_types:
    data_with_irrelevant[label] = 0
for i in range(len(data)):
    for label in data_with_irrelevant.iloc[i]["labels"]:
        data_with_irrelevant.at[i, label] = 1

# # binary labels for each row
for label in label_types:
    data[label] = 0
for i in range(len(data)):
    for label in data.iloc[i]["labels"]:
        data.at[i, label] = 1

ukrainian_usernames = ["MFA_Ukraine", "UKRinUN", "Ukraine", "DefenceU", "UKRintheUSA", "DmytroKuleba", "oleksiireznikov", "SergiyKyslytsya", "Denys_Shmyhal", "OlegNikolenko_", "EmineDzheppar", "ZelenskyyUa"]
individual_usernames = ["DmytroKuleba", "oleksiireznikov", "SergiyKyslytsya", "Denys_Shmyhal", "OlegNikolenko_", "EmineDzheppar", "ZelenskyyUa", "Dpol_un", "MedvedevRussiaE"]

data["ukrainian"] = data["username"].apply(lambda x: 1 if x in ukrainian_usernames else 0)
data["individual"] = data["username"].apply(lambda x: 1 if x in individual_usernames else 0)


# # Irrelevant data analysis

# irrelevant_data = data_with_irrelevant[data_with_irrelevant["relevant"] == False]

# print(len(irrelevant_data))
# print(sum(irrelevant_data["ukrainian"]))
# print(sum(irrelevant_data["individual"]))

# grouped_irrelevant = irrelevant_data.groupby("username")["labels"].describe().reset_index().sort_values("count", ascending=False).reset_index(drop=True)

# other_ukraine = 0
# other_russia = 0
# for i in range(5, len(grouped_irrelevant)):
#     if grouped_irrelevant["username"][i] in ukrainian_usernames:
#         other_ukraine += grouped_irrelevant["count"][i]
#     else:
#         other_russia += grouped_irrelevant["count"][i]

# print(other_ukraine, other_russia)

# x = grouped_irrelevant["username"][:5].tolist() + ["Other Russian", "Ukrainian"]
# y = grouped_irrelevant["count"][:5].tolist() + [other_russia, other_ukraine]

# print(data_with_irrelevant.columns)
# mfa = data_with_irrelevant[data_with_irrelevant["username"] == "mfa_russia"][["text_without_links", "photo_link"]].iloc[1]
# un = irrelevant_data[irrelevant_data["username"] == "RussiaUN"][["text_without_links", "photo_link"]].iloc[0]
# ukraine = irrelevant_data[irrelevant_data["ukrainian"] == 1][["text_without_links", "photo_link"]].iloc[2]
# irrelevant_plot = [mfa, un, ukraine]

# for i, e in enumerate(irrelevant_plot):
#     print(e)

# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# plt.subplots_adjust(bottom=0.25)
# for i, e in enumerate(irrelevant_plot):
#     ax[i].imshow(e[1])
#     ax[i].annotate(e[0], (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top', wrap=True)
#     # ax[i].annotate(e[0],  wrap=True)
#     ax[i].tick_params(
#         axis='both',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         left=False,
#         right=False,
#         labelleft=False,
#         labelbottom=False) # labels along the bottom edge are off
# plt.show()





# fig, ax = plt.subplots()
# plt.bar(x, y, color=["#C02657", "#C02657", "#C02657", "#C02657", "#C02657", "#F1437D", "#1F6ABF"])
# ax.set_ylabel("Number of irrelevant tweets")
# ax.bar_label(ax.containers[0], label_type="edge")
# plt.xticks(rotation=25)
# plt.savefig("graphs/irrelevant_tweets.pdf", bbox_inches='tight')
# # plt.show()

# mask = data.labels.apply(lambda x: 'Portrait' in x)
# portrait_data = data[mask]
# mask = data.labels.apply(lambda x: 'Text' in x)
# text_data = data[mask]

# print(len(portrait_data))
# print(sum(portrait_data["ukrainian"]))
# print(sum(portrait_data["individual"]))

# print(len(text_data))
# print(sum(text_data["ukrainian"]))
# print(sum(text_data["individual"]))

# grouped_portrait = portrait_data.groupby("username")["labels"].describe().reset_index().sort_values("count", ascending=False).reset_index(drop=True)
# grouped_text = text_data.groupby("username")["labels"].describe().reset_index().sort_values("count", ascending=False).reset_index(drop=True)

# print(grouped_portrait)
# print(grouped_text)

# x = grouped_portrait["username"][:5].tolist() + ["Other"]
# y = grouped_portrait["count"][:5].tolist() + [sum(grouped_portrait["count"][5:])]
# x = grouped_text["username"][:5].tolist() + ["Other"]
# y = grouped_text["count"][:5].tolist() + [sum(grouped_text["count"][5:])]
# fig, ax = plt.subplots()
# plt.bar(x, y, color=["#1F6ABF", "#1F6ABF", "#C02657", "#C02657", "#1F6ABF", "#3CC130"])
# ax.set_ylabel("Number of texts in images")
# ax.bar_label(ax.containers[0], label_type="edge")
# red_patch = mpatches.Patch(color='#C02657', label='Russian')
# blue_patch = mpatches.Patch(color='#1F6ABF', label='Ukrainian')
# green_patch = mpatches.Patch(color='#3CC130', label='Other')
# ax.legend(handles=[red_patch, blue_patch])
# plt.xticks(rotation=25)
# plt.savefig("graphs/text_tweets.pdf", bbox_inches='tight')
# plt.show()


# data["ukrainian"] = data["username"].apply(lambda x: 1 if x in ukrainian_usernames else 0)
# data["individual"] = data["username"].apply(lambda x: 1 if x in individual_usernames else 0)

data_without_individuals = data[data["individual"] != 1].reset_index()
data_without_russians = data[data["ukrainian"] == 1].reset_index()
data_without_ukrainians = data[data["ukrainian"] == 0].reset_index()
# data_without_ukrainians["west"] = data_without_ukrainians["text_without_links"].apply(lambda x: 1 if "west" in x.lower() else 0)

print(len(data_without_ukrainians))
# print(sum(data_without_ukrainians["text_without_links"].str.contains("West|west")))
# print(sum(data_without_ukrainians["text_without_links"].str.contains("NATO|nato|Nato")))
# print(sum(data_without_ukrainians["text_without_links"].str.contains("Regime|regime")))
# print(data_without_ukrainians[data_without_ukrainians["Doubt"] == 1]["text_without_links"].tolist())
# print(len(data_without_russians))
# print(sum(data_without_russians["text_without_links"].str.contains("Terrorist|terrorist")))
# print(sum(data_without_russians["text_without_links"].str.contains("Occupier|occupier")))
print(sum(data_without_russians["text_without_links"].str.contains("russia")))

# print(data_without_russians[data_without_russians["text_without_links"].str.contains("orc|Orc")]["text_without_links"])
# print(data_without_ukrainians[data_without_ukrainians["text_without_links"].str.contains("US")]["text_without_links"])

# print(len(data_without_individuals[data_without_individuals["ukrainian"] == 0]))

# metrics = td.extract_metrics(
#     text = data["text_without_links"],
#     lang = "en",
#     metrics = None
# )

# metrics_df = data.join(metrics.drop(columns=["text"]))
# # # print(metrics_df.sort_values("sentence_length_mean", ascending=False).head(5).iloc[0]["text_without_links"])

# corr_results_df = pd.DataFrame()
# columns = []
# corrs = []
# p_values = []
# metrics_df.replace(np.nan, 0, inplace=True)
# for column in metrics_df.columns:
#     if column != "ukrainian" and column not in data.columns:
#         columns.append(column)
#         corr, p_value = spearmanr(metrics_df[column], metrics_df["ukrainian"])
#         corrs.append(corr)
#         p_values.append(p_value)

# corr_results_df['Feature'] = columns
# corr_results_df['Correlation'] = corrs
# corr_results_df['p_value'] = p_values

# corr_results_df.sort_values(by="Correlation", key=abs, ascending=False, inplace=True)
# print(corr_results_df[:10])
# metrics_correlations = metrics.corrwith(metrics_df["ukrainian"], method="spearman").sort_values(key=abs, ascending=False)

# print(metrics_correlations[:10])

# fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
# for i, metric in enumerate(["n_characters", "per_word_perplexity", "rix"]):
#     sns.kdeplot(metrics_df, x=metric, hue="ukrainian", palette=["#1F6ABF", "#C02657"], ax=ax[i], common_norm=False)
#     ax[i].legend(labels=["Russian", "Ukrainian"])

# ax[0].set_xlabel("Number of characters")
# ax[1].set_xlabel("Per word perplexity")
# ax[2].set_xlabel("Rix")
# # # plt.xlabel("Number of characters")
# # plt.savefig("graphs/individual_organization_textdescriptives.pdf")
# plt.show()
grouping_data = data.groupby(["username"]).apply(lambda x: x[label_types].sum()/len(x)).reset_index()
low_tweets = ["mission_rf", "RusMission_EU", "natomission_ru", "Dpol_un", "MedvedevRussiaE", "UKRinUN", "Ukraine",
              "UKRintheUSA", "OlegNikolenko_", "ZelenskyyUa"]
grouping_data.drop(grouping_data[grouping_data.username.isin(low_tweets)].index, inplace=True)


grouping_data["ukrainian"] = grouping_data["username"].apply(lambda x: 1 if x in ukrainian_usernames else 0)
grouping_data["individual"] = grouping_data["username"].apply(lambda x: 1 if x in individual_usernames else 0)

# # grouping_ukrainian_data = grouping_data[grouping_data["ukrainian"] == 1].reset_index()
# # # grouping_russian_data = grouping_data[grouping_data["ukrainian"] == 0]
# # # grouping_organization_data = grouping_data[grouping_data["individual"] == 0]

# # # # print(grouping_data)
# summarized = grouping_data.groupby(["ukrainian"]).describe().reset_index()
# print(summarized)

# for label in label_types:
# #     ttest = pg.ttest(grouping_ukrainian_data[grouping_ukrainian_data["individual"] == 0][label], grouping_ukrainian_data[grouping_ukrainian_data["individual"] == 1][label], correction=True)
#     ttest = pg.ttest(grouping_data[grouping_data["ukrainian"] == 0][label], grouping_data[grouping_data["ukrainian"] == 1][label], correction=True)
# # print(ttest)

#     if ttest["p-val"]["T-test"] < 0.05:
#         print(ttest, label)

# for label in label_types:
#     print(shapiro(grouping_ukrainian_data[label]), label)


grouping_data["Nationality"] = grouping_data["ukrainian"].apply(lambda x: "Russian" if x == 0 else "Ukrainian")
fig, ax = plt.subplots()#1, 3, figsize=(15, 5), sharey=False, sharex=True)

points = ["Appeal to fear"]

print(grouping_data.sort_values(by=points[0]))

# for i, e in enumerate(points):
#     sns.boxplot(x="Nationality", y=e, data=grouping_data, palette=["#C02657", "#1F6ABF"], boxprops={'alpha': 0.4}, ax=ax, showfliers=False)
#     sns.stripplot(x="Nationality", y=e, data=grouping_data, palette=["#C02657", "#1F6ABF"], ax=ax)
#     ax.set(ylabel=e + " frequency")
# plt.savefig("graphs/fear.pdf", bbox_inches='tight')
# plt.show()








# # fig, ax = plt.subplots()
# low_tweets = ["mission_rf", "RusMission_EU", "natomission_ru", "Dpol_un", "MedvedevRussiaE", "UKRinUN", "Ukraine",
#               "UKRintheUSA", "OlegNikolenko_", "ZelenskyyUa"]

# impression_data = data.drop(data[data.username.isin(low_tweets)].index)
# impression_data.drop(impression_data[impression_data["impression_count"] == 0].index, inplace=True)

# impression_medians = impression_data.groupby(["username"]).describe()["impression_count"]["50%"]


# impression_data["impression_difference"] = impression_data["username"].apply(lambda x: impression_medians[x])

# # impression_data["impression_difference"] = (impression_data["impression_count"] - impression_data["impression_difference"]) / impression_data["impression_difference"]
# impression_data["impression_difference"] = np.array(impression_data["impression_count"]) >= np.array(impression_data["impression_difference"])


# impression_data["impression_difference"] = impression_data["impression_difference"].apply(lambda x: 0 if x is False else 1)
# # pd.cut(impression_data.impression_difference, bins=4, labels=np.arange(4), right=False)


# X_train, X_test, y_train, y_test = train_test_split(impression_data[label_types], impression_data["impression_difference"], random_state=99)

# regr = LogisticRegression()
# regr.fit(X_train, y_train)

# f_values, p_values = f_classif(impression_data[label_types], impression_data["impression_difference"])
# biggest = np.argsort(f_values)

# y_pred = regr.predict(X_test)
# importance = regr.coef_[0]
# for i in biggest:
#     print(f_values[i], p_values[i], label_types[i], importance[i])

# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print("ROC_AUC: %.2f" % roc_auc_score(y_test, y_pred))

# sample_weights = np.empty(len(y_train))
# sample_weights[y_train == 0] = sum(y_train) / len(y_train)
# sample_weights[y_train == 1] = 1 - sum(y_train) / len(y_train)

# ebm = ExplainableBoostingClassifier()
# ebm.fit(X_train, y_train, sample_weight=sample_weights)

# ebm_terms = ebm.term_names_
# ebm_scores = ebm.term_scores_
# ebm_importances = ebm.term_importances()

# ebm_biggest = np.argsort(ebm_importances)

# for i in ebm_biggest:
#     print(ebm_scores[i], ebm_importances[i], ebm_terms[i])

# show(ebm.explain_global())

# y_pred = ebm.predict(X_test)
# print(mean_squared_error(y_test, y_pred))
# print(roc_auc_score(y_test, y_pred))

# print(sum(y_test) - sum(y_pred))

# for i in range(len(data)):
#     if "Plain folks" in data.iloc[i]["labels"]:
#         print(data.iloc[i]["text_without_links"])



# PHOTO DOWNLOADING CHUNK
# urls = data.photo_link.tolist()
# for url in urls:
#     img_data = requests.get(url).content
#     photo = url.rsplit('/')[-1]
#     with open(f"photo_links/{photo}", 'wb') as handler:
#         handler.write(img_data)

# plt.figure(dpi=1200)
# fig, ax = plt.subplots()


# # datelist = data["created_at"].tolist()
# # converted_dates = list(map(datetime.datetime.strptime, datelist, len(datelist)*['%Y-%m-%dT%H:%M:%S.%fZ']))
# # x_axis = converted_dates
# # formatter = dates.DateFormatter('%Y-%m-%d')

# plt.bar(occurences.keys(), occurences.values(), color="#1F6ABF")
# ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# ax.set_xlabel("Number of distinct propaganda techniques in a tweet")
# ax.set_ylabel("Number of tweets")
# ax.bar_label(ax.containers[0], label_type="edge")
# # plt.hist(x_axis, bins=100, color="#1F6ABF")
# # ax = plt.gcf().axes[0] 
# # ax.xaxis.set_major_formatter(formatter)
# # ax.set_ylabel("Number of tweets")
# # ax.set_xlabel("Date")


# # plt.gcf().autofmt_xdate(rotation=25)

# plt.savefig("graphs/occurences.pdf")

# plt.show()
# %%
