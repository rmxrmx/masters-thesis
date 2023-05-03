import pandas as pd
import numpy as np
import glob
import ast
import re

all_files = glob.glob("csvs/full*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

print(frame["created_at"].sort_values())

link_regex = re.compile(r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|"""
                    r"""www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?"""
                    r""":[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))"""
                    r"""*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|"""
                    r"""[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")

frame["media_type"] = frame["attachments.media"].apply(lambda x: ast.literal_eval(x)[0]["type"] if x is not np.nan else np.nan)
frame["photo_link"] = frame["attachments.media"].apply(lambda x: ast.literal_eval(x)[0]["url"] if x is not np.nan and ast.literal_eval(x)[0]["type"] == "photo" else np.nan)
frame["text_without_links"] = frame["text"].apply(lambda x: link_regex.sub("", x) if x is not np.nan else np.nan)

useful_columns = ["id", "author_id", "created_at", "lang",
                  "public_metrics.impression_count", "public_metrics.reply_count", "public_metrics.retweet_count",
                  "public_metrics.quote_count", "public_metrics.like_count", "entities.annotations",
                  "context_annotations", "author.username", "author.name", "author.description",
                  "author.public_metrics.followers_count", "media_type", "text_without_links", "photo_link"]

updated_df = frame[useful_columns]
updated_df = updated_df[updated_df["media_type"] == "photo"]
updated_df = updated_df[updated_df["lang"] == "en"]
# updated_df = updated_df[updated_df["text_without_links"] != np.nan]
print(len(updated_df))

# updated_df.to_csv("csvs/formatted_data.csv")