#!/usr/bin/env python3

import time
import pandas as pd
import nltk
import string
import percache
import re
import Levenshtein
import jellyfish
from nltk.corpus import stopwords
from pandas.core.series import Series
from sklearn.grid_search import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from dictionary import brand_typos, typos, abbreviations


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), " ") for char in string.punctuation)

SEED = 837

CV_N_FOLDS = 3

TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"
ATTRIBUTES_CSV = "./data/attributes.csv"
DESCRIPTIONS_CSV = "./data/product_descriptions.csv"
CACHE_DIR = "./cache"
TRAIN_CACHE = "%s/train.csv" % CACHE_DIR
TEST_CACHE = "%s/test.csv" % CACHE_DIR
MODEL_CACHE = "%s/xgb.pkl" % CACHE_DIR
PREDS_CACHE = "%s/predictions.csv" % CACHE_DIR


def myrepr(arg):
    """Custom implementation of 'myrepr' function

    The default one doesn't handle complex types like Pandas' Series or TfidfVectorizer
    """
    if isinstance(arg, pd.core.series.Series):
        return "%s_%s" % (str(arg.shape), str(arg.name))
    if isinstance(arg, TfidfVectorizer):
        return "vectorizer"
    else:
        return repr(arg)

cache = percache.Cache("%s/percache" % CACHE_DIR, repr=myrepr)


def rmse(y, y_pred):
    """Root Mean Squared Error function

    See: https://www.kaggle.com/wiki/RootMeanSquaredError
    """
    return mean_squared_error(y, y_pred) ** 0.5


@cache
def word_len(df, feature):
    """Counts the number of words in given feature"""
    return df[feature].map(lambda x: len(x.split()))


@cache
def char_len(df, feature):
    """Counts the number of characters in given feature"""
    return df[feature].map(lambda x: len(x))


@cache
def search_term_in_title(df):
    def calc(row):
        return len(list(set(row["search_term_stem"].split()).intersection(row["product_title_stem"].split())))
    return df.apply(calc, axis=1)


@cache
def search_term_in_title_ratio(df):
    def calc(row):
        if row["search_term_len"] == 0:
            return 0
        return row["search_term_in_product_title"] / row["search_term_len"]
    return df.apply(calc, axis=1)


@cache
def search_term_in_description(df):
    def calc(row):
        return len(list(set(row["search_term_stem"].split()).intersection(row["product_description_stem"].split())))
    return df.apply(calc, axis=1)


@cache
def search_term_in_description_ratio(df):
    def calc(row):
        if row["search_term_len"] == 0:
            return 0
        return row["search_term_in_product_description"] / row["search_term_len"]
    return df.apply(calc, axis=1)


@cache
def search_term_in_brand(df):
    def calc(row):
        if type(row["brand"]) is str:
            return len(list(set(row["search_term_stem"].split()).intersection(row["brand_stem"].split())))
        else:
            return 0
    return df.apply(calc, axis=1)


@cache
def search_term_in_brand_ratio(df):
    def calc(row):
        if row["search_term_len"] == 0:
            return 0
        return row["search_term_in_brand"] / row["search_term_len"]
    return df.apply(calc, axis=1)


@cache
def search_term_in_material(df):
    def calc(row):
        if type(row["material"]) is str:
            return len(list(set(row["search_term_stem"].split()).intersection(row["material_stem"].split())))
        else:
            return 0
    return df.apply(calc, axis=1)


@cache
def search_term_in_material_ratio(df):
    def calc(row):
        if row["search_term_len"] == 0:
            return 0
        return row["search_term_in_material"] / row["search_term_len"]
    return df.apply(calc, axis=1)


@cache
def search_term_in_color_family(df):
    def calc(row):
        if type(row["color_family"]) is str:
            return len(list(set(row["search_term_stem"].split()).intersection(row["color_family_stem"].split())))
        else:
            return 0
    return df.apply(calc, axis=1)


@cache
def search_term_in_color_family_ratio(df):
    def calc(row):
        if row["search_term_len"] == 0:
            return 0
        return row["search_term_in_color_family"] / row["search_term_len"]
    return df.apply(calc, axis=1)


@cache
def tfidf_corpus(corpus, tfidf_vectorizer):
    return tfidf_vectorizer.fit_transform(corpus), tfidf_vectorizer


@cache
def cosine_sim(dataframe, tfidf_vectorizer):
    def vectorize_and_calculate_distance(row):
        cos = cosine_similarity(tfidf_vectorizer.transform([row.iloc[0]]), tfidf_vectorizer.transform([row.iloc[1]]))
        return cos[0][0]
    return dataframe.apply(vectorize_and_calculate_distance, axis=1)


@cache
def jaccard_distance(dataframe, feature1, feature2):
    """Calculates the Jaccard distance between two features"""
    def calculate_jaccard(row):

        if type(row[feature1]) is not str or type(row[feature2]) is not str:
            return 0

        feature1_set = set(row[feature1].split())
        feature2_set = set(row[feature2].split())

        intersection = feature1_set.intersection(feature2_set)
        union = feature1_set.union(feature2_set)
        if len(union) == 0:
            return 0
        return len(intersection) / len(union)
    return dataframe.apply(calculate_jaccard, axis=1)


@cache
def levenshtein_distance(dataframe, feature1, feature2):
    """Calculates the Levenshtein (edit) distance between two features"""
    def calculate_levenshtein(row):

        if type(row[feature1]) is not str or type(row[feature2]) is not str:
            return -1

        return Levenshtein.distance(row[feature1], row[feature2])
    return dataframe.apply(calculate_levenshtein, axis=1)


@cache
def jaro_distance(dataframe, feature1, feature2):
    """Calculates the Jaro distance between two features"""
    def calculate_jaro(row):

        if type(row[feature1]) is not str or type(row[feature2]) is not str:
            return -1

        return jellyfish.jaro_distance(row[feature1], row[feature2])
    return dataframe.apply(calculate_jaro, axis=1)


def spellcheck(sentence, dictionary):
    """TODO: see str().translate() for a probably better implementation"""

    words = list()

    for word in sentence.split():
        if word in dictionary.keys():
            words.append(dictionary[word])
        else:
            words.append(word)

    return " ".join(words)


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]


def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


def build_tfidf(df_train, df_test):

    vectorizer = TfidfVectorizer(tokenizer=normalize)

    tfidf = dict()

    df = pd.concat([df_train, df_test])

    print("Building TF-IDF vocabulary for product title...", end="", flush=True)
    t1 = time.time()
    (tfidf_product_title, vectorizer_product_title) = tfidf_corpus(df["product_title_stem"], vectorizer)
    print("Done (%d secs)" % (time.time() - t1))

    print("Building TF-IDF vocabulary for product description...", end="", flush=True)
    t1 = time.time()
    (tfidf_product_description, vectorizer_product_description) = tfidf_corpus(df["product_description_stem"], vectorizer)
    print("Done (%d secs)" % (time.time() - t1))

    print("Building TF-IDF vocabulary for brand...", end="", flush=True)
    t1 = time.time()
    (tfidf_brand, vectorizer_brand) = tfidf_corpus(df["brand_stem"], vectorizer)
    print("Done (%d secs)" % (time.time() - t1))

    print("Building TF-IDF vocabulary for material...", end="", flush=True)
    t1 = time.time()
    (tfidf_material, vectorizer_material) = tfidf_corpus(df["material_stem"], vectorizer)
    print("Done (%d secs)" % (time.time() - t1))

    tfidf["product_title"] = {
        "vocabulary": tfidf_product_title,
        "vectorizer": vectorizer_product_title
    }

    tfidf["product_description"] = {
        "vocabulary": tfidf_product_description,
        "vectorizer": vectorizer_product_description
    }

    tfidf["brand"] = {
        "vocabulary": tfidf_brand,
        "vectorizer": vectorizer_brand
    }

    tfidf["material"] = {
        "vocabulary": tfidf_material,
        "vectorizer": vectorizer_material
    }

    return tfidf


def merge_dataframe(df, df_descriptions, df_attributes):
    """Merges attributes and descriptions into the main dataset"""

    # There are duplicate entries for the material attribute. I.e.:
    # 100563,"Material","Stainless steel"
    # 100563, "Material", "Stainless Steel"
    df_attributes.drop_duplicates(subset=("product_uid", "name"), inplace=True)

    print("Merging product description... ", end="", flush=True)
    t1 = time.time()
    df = pd.merge(df, df_descriptions, how="left", on="product_uid")
    print("Done (%d secs)" % (time.time() - t1))

    print("Merging brand... ", end="", flush=True)
    t1 = time.time()
    df = pd.merge(df, df_attributes[df_attributes.name == "MFG Brand Name"], how="left", on="product_uid")
    df.drop("name", axis=1, inplace=True)
    df.rename(columns={"value": "brand"}, inplace=True)
    print("Done (%d secs)" % (time.time() - t1))

    print("Merging material... ", end="", flush=True)
    t1 = time.time()
    df = pd.merge(df, df_attributes[df_attributes.name == "Material"], how="left", on="product_uid")
    df.drop("name", axis=1, inplace=True)
    df.rename(columns={"value": "material"}, inplace=True)
    print("Done (%d secs)" % (time.time() - t1))

    print("Merging color family... ", end="", flush=True)
    t1 = time.time()
    df = pd.merge(df, df_attributes[df_attributes.name == "Color Family"], how="left", on="product_uid")
    df.drop("name", axis=1, inplace=True)
    df.rename(columns={"value": "color_family"}, inplace=True)
    print("Done (%d secs)" % (time.time() - t1))

    print("Merging accessory... ", end="", flush=True)
    t1 = time.time()
    df = pd.merge(df, df_attributes[df_attributes.name == "Accessory Type"], how="left", on="product_uid")
    df.drop("name", axis=1, inplace=True)
    df.rename(columns={"value": "accessory"}, inplace=True)
    df.accessory.fillna("0", inplace=True)
    df.accessory = df.accessory.map(lambda x: 0 if x == 0 else 1)
    print("Done (%d secs)" % (time.time() - t1))

    return df


@cache
def clean_data(df):
    """Data cleaning operations"""

    print("Cleaning titles... ", end="", flush=True)
    t1 = time.time()
    df.product_title = df.product_title.map(lambda x: x.lower())
    df.product_title = df.product_title.map(lambda x: x.replace("&amp;", "&"))
    df.product_description = df.product_description.map(lambda x: x.replace("&nbsp;", " "))
    df.product_title = df.product_title.map(lambda x: x.replace("&#39;", "'"))
    df.product_title = df.product_title.map(lambda x: x.replace("-DISCONTINUED", ""))
    print("Done (%d secs)" % (time.time() - t1))

    print("Cleaning search terms... ", end="", flush=True)
    t1 = time.time()

    df.search_term = df.search_term.map(lambda x: x.lower())

    # Expliciting acronymns
    df.search_term = df.search_term.map(lambda x: spellcheck(x, abbreviations))

    # IMPORTANT
    # see for synonyms! Like 'light' in search_term and 'Lantern' in product title!
    # or related! Like 'wood' in search_term and 'pine' in product title!
    # or related! Like 'steel' in search_term and 'metal' in product title!
    df.search_term = df.search_term.map(lambda x: x.replace(" cement ", " concrete "))

    # Fixing brands
    df.search_term = df.search_term.map(lambda x: spellcheck(x, brand_typos))

    # Expanding abbreviations
    df.search_term = df.search_term.map(lambda x: x.replace(" qtr ", " quarter "))
    df.search_term = df.search_term.map(lambda x: x.replace(" vac ", " vacuum "))
    df.search_term = df.search_term.map(lambda x: x.replace(" ac ", " air conditioner "))
    df.search_term = df.search_term.map(lambda x: x.replace(" vert ", " vertical "))
    df.search_term = df.search_term.map(lambda x: x.replace(" sld ", " solid "))

    # Translations
    df.search_term = df.search_term.map(lambda x: x.replace("aspiradora", "vacuum"))
    df.search_term = df.search_term.map(lambda x: x.replace("aire acondicionado", "air condicioner"))

    # Spellchecking (raw...)
    df.search_term = df.search_term.map(lambda x: spellcheck(x, typos))

    # Normalizing units of measure
    df.search_term = df.search_term.map(lambda x: re.sub(r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt ", x))
    df.search_term = df.search_term.map(lambda x: re.sub(r"([0-9]+)( *)(pounds|pound|lb|lbs)\.?", r"\1 lb ", x))
    df.search_term = df.search_term.map(lambda x: re.sub(r"([0-9]+)( *)(oz)\.?", r"\1 oz ", x))
    df.search_term = df.search_term.map(lambda x: re.sub(r"([0-9]+)( *)(gallon|gallons)\.?", r"\1 gal ", x))

    # Watts are not following a convention even in product titles (60W/60-Watts)
    df.search_term = df.search_term.map(lambda x: re.sub(r"([0-9]+)( *)(watts|watt|wattsolar)\.?", r"\1 watt ", x))
    df.search_term = df.search_term.map(lambda x: re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp ", x))
    df.search_term = df.search_term.map(lambda x: re.sub(r"([0-9]+)( *)(foot|feet|ft)\.?", r"\1 ft", x))
    df.search_term = df.search_term.map(lambda x: re.sub(r"([0-9]+)x([0-9]+)\.?", r"\1 x \2", x))

    # We must find a solution for 'in.' (inches) that after removal of punctuation is removed by
    # the tokenizer as being an english stopword!
    df.search_term = df.search_term.map(lambda x: x.replace("inch", "in."))

    # # Splitting words
    # # Something must be done for words like this: outdoorlounge/heavyduty
    print("Done (%d secs)" % (time.time() - t1))

    print("Cleaning product descriptions... ", end="", flush=True)
    t1 = time.time()
    df.product_description = df.product_description.map(lambda x: x.lower())
    df.product_description = df.product_description.map(lambda x: x.replace("&amp;", "&"))
    df.product_description = df.product_description.map(lambda x: x.replace("&nbsp;", " "))
    df.product_description = df.product_description.map(lambda x: x.replace("&#39;", "'"))
    print("Done (%d secs)" % (time.time() - t1))

    print("Cleaning brands... ", end="", flush=True)
    t1 = time.time()
    df.brand.fillna("", inplace=True)
    df.brand = df.brand.map(lambda x: x.lower())
    print("Done (%d secs)" % (time.time() - t1))

    print("Cleaning material... ", end="", flush=True)
    t1 = time.time()
    df.material.fillna("", inplace=True)
    df.material = df.material.map(lambda x: x.lower())
    print("Done (%d secs)" % (time.time() - t1))

    print("Cleaning color_family... ", end="", flush=True)
    t1 = time.time()
    df.color_family.fillna("", inplace=True)
    df.color_family = df.color_family.map(lambda x: x.lower())
    print("Done (%d secs)" % (time.time() - t1))

    return df


@cache
def tokenize_data(df):

    stop_words = set(stopwords.words('english'))

    print("Tokenizing search term... ", end="", flush=True)
    t1 = time.time()
    df["search_term_token"] = df["search_term"].map(
        lambda x: ' ' .join(nltk.word_tokenize(x.translate(remove_punctuation_map))))
    df["search_term_token"] = df["search_term_token"].map(
        lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    print("Done (%d secs)" % (time.time() - t1))

    print("Tokenizing product title... ", end="", flush=True)
    t1 = time.time()
    df["product_title_token"] = df["product_title"].map(
        lambda x: ' '.join(nltk.word_tokenize(x.translate(remove_punctuation_map))))
    df["product_title_token"] = df["product_title_token"].map(
        lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    print("Done (%d secs)" % (time.time() - t1))

    print("Tokenizing product description... ", end="", flush=True)
    t1 = time.time()
    df["product_description_token"] = df["product_description"].map(
        lambda x: ' '.join(nltk.word_tokenize(x.translate(remove_punctuation_map))))
    df["product_description_token"] = df["product_description_token"].map(
        lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    print("Done (%d secs)" % (time.time() - t1))

    print("Tokenizing brand... ", end="", flush=True)
    t1 = time.time()
    df["brand_token"] = df["brand"].map(
        lambda x: ' '.join(nltk.word_tokenize(x.translate(remove_punctuation_map))))
    df["brand_token"] = df["brand_token"].map(
        lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    print("Done (%d secs)" % (time.time() - t1))

    print("Tokenizing material... ", end="", flush=True)
    t1 = time.time()
    df["material_token"] = df["material"].map(
        lambda x: ' '.join(nltk.word_tokenize(x.translate(remove_punctuation_map))))
    df["material_token"] = df["material_token"].map(
        lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    print("Done (%d secs)" % (time.time() - t1))

    print("Tokenizing color_family... ", end="", flush=True)
    t1 = time.time()
    df["color_family_token"] = df["color_family"].map(
        lambda x: ' '.join(nltk.word_tokenize(x.translate(remove_punctuation_map))))
    df["color_family_token"] = df["color_family_token"].map(
        lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    print("Done (%d secs)" % (time.time() - t1))

    print("Caching debug file to %s" % ("./cache/%s_debug_token.csv" % len(df)))
    df.to_csv("./cache/%s_debug_token.csv" % len(df), encoding="UTF-8", index=False)
    print("Done (%d secs)" % (time.time() - t1))

    return df


@cache
def stem_data(df):

    print("Stemming search term... ", end="", flush=True)
    t1 = time.time()
    df["search_term_stem"] = df["search_term_token"].map(lambda x: ' '.join([stemmer.stem(item) for item in x.split()]))
    print("Done (%d secs)" % (time.time() - t1))

    print("Stemming product title... ", end="", flush=True)
    t1 = time.time()
    df["product_title_stem"] = df["product_title_token"].map(
        lambda x: ' '.join([stemmer.stem(item) for item in x.split()]))
    print("Done (%d secs)" % (time.time() - t1))

    print("Stemming product description... ", end="", flush=True)
    t1 = time.time()
    df["product_description_stem"] = df["product_description_token"].map(
        lambda x: ' '.join([stemmer.stem(item) for item in x.split()]))
    print("Done (%d secs)" % (time.time() - t1))

    print("Stemming brand... ", end="", flush=True)
    t1 = time.time()
    df["brand_stem"] = df["brand_token"].map(lambda x: ' '.join([stemmer.stem(item) for item in x.split()]))
    print("Done (%d secs)" % (time.time() - t1))

    print("Stemming material... ", end="", flush=True)
    t1 = time.time()
    df["material_stem"] = df["material_token"].map(lambda x: ' '.join([stemmer.stem(item) for item in x.split()]))
    print("Done (%d secs)" % (time.time() - t1))

    print("Stemming color_family... ", end="", flush=True)
    t1 = time.time()
    df["color_family_stem"] = df["color_family_token"].map(lambda x: ' '.join([stemmer.stem(item) for item in x.split()]))
    print("Done (%d secs)" % (time.time() - t1))

    print("Caching debug file to %s" % ("./cache/%s_debug_stem.csv" % len(df)))
    df.to_csv("./cache/%s_debug_stem.csv" % len(df), encoding="UTF-8", index=False)
    print("Done (%d secs)" % (time.time() - t1))

    return df


@cache
def feature_engineering(df, tfidf):
    """Various feature engineering operations:

    - Cosine similarity
    - Words in query
    - Words in query ratio
    """
    
    print("Calculating cosine similarity between search_term and product_title... ", end="", flush=True)
    t1 = time.time()
    df["cs_search_term_product_title"] = cosine_sim(
        df[["search_term_stem", "product_title_stem"]], tfidf["product_title"]["vectorizer"])
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating cosine similarity between search_term and product_description... ", end="", flush=True)
    t1 = time.time()
    df["cs_search_term_product_description"] = cosine_sim(
        df[["search_term_stem", "product_description_stem"]], tfidf["product_description"]["vectorizer"])
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating cosine similarity between search_term and brand... ", end="", flush=True)
    t1 = time.time()
    df["cs_search_term_brand"] = cosine_sim(df[["search_term_stem", "brand_stem"]], tfidf["brand"]["vectorizer"])
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating cosine similarity between search_term and material... ", end="", flush=True)
    t1 = time.time()
    df["cs_search_term_material"] = cosine_sim(
        df[["search_term_stem", "material_stem"]], tfidf["material"]["vectorizer"])
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search term length... ", end="", flush=True)
    t1 = time.time()
    df["search_term_len"] = word_len(df, "search_term_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating product title length... ", end="", flush=True)
    t1 = time.time()
    df["product_title_len"] = word_len(df, "product_title_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating product description length... ", end="", flush=True)
    t1 = time.time()
    df["product_description_len"] = word_len(df, "product_description_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating brand length... ", end="", flush=True)
    t1 = time.time()
    df["brand_len"] = word_len(df, "brand_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating material length... ", end="", flush=True)
    t1 = time.time()
    df["material_len"] = word_len(df, "material_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating color_family length... ", end="", flush=True)
    t1 = time.time()
    df["color_family_len"] = word_len(df, "color_family_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search term length (in chars)... ", end="", flush=True)
    t1 = time.time()
    df["search_term_len_char"] = char_len(df, "search_term_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating product title length (in chars)... ", end="", flush=True)
    t1 = time.time()
    df["product_title_len_char"] = char_len(df, "product_title_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating product description length (in chars)... ", end="", flush=True)
    t1 = time.time()
    df["product_description_len_char"] = char_len(df, "product_description_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating brand length (in chars)... ", end="", flush=True)
    t1 = time.time()
    df["brand_len_char"] = char_len(df, "brand_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating material length (in chars)... ", end="", flush=True)
    t1 = time.time()
    df["material_len_char"] = char_len(df, "material_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating color_family length (in chars)... ", end="", flush=True)
    t1 = time.time()
    df["color_family_len_char"] = char_len(df, "color_family_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search terms in product title... ", end="", flush=True)
    t1 = time.time()
    df["search_term_in_product_title"] = search_term_in_title(df)
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search terms / product title ratio... ", end="", flush=True)
    t1 = time.time()
    df["search_term_in_product_title_ratio"] = search_term_in_title_ratio(df)
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search terms in product description... ", end="", flush=True)
    t1 = time.time()
    df["search_term_in_product_description"] = search_term_in_description(df)
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search terms / product description ratio... ", end="", flush=True)
    t1 = time.time()
    df["search_term_in_product_description_ratio"] = search_term_in_description_ratio(df)
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search terms in brand... ", end="", flush=True)
    t1 = time.time()
    df["search_term_in_brand"] = search_term_in_brand(df)
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search terms / brand ratio... ", end="", flush=True)
    t1 = time.time()
    df["search_term_in_brand_ratio"] = search_term_in_brand_ratio(df)
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search terms in material... ", end="", flush=True)
    t1 = time.time()
    df["search_term_in_material"] = search_term_in_material(df)
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search terms / material ratio... ", end="", flush=True)
    t1 = time.time()
    df["search_term_in_material_ratio"] = search_term_in_material_ratio(df)
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search terms in color_family... ", end="", flush=True)
    t1 = time.time()
    df["search_term_in_color_family"] = search_term_in_color_family(df)
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating search terms / color_family ratio... ", end="", flush=True)
    t1 = time.time()
    df["search_term_in_color_family_ratio"] = search_term_in_color_family_ratio(df)
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Jaccard distance between search_term and product_title... ", end="", flush=True)
    t1 = time.time()
    df["jaccard_search_term_product_title"] = jaccard_distance(df, "search_term_stem", "product_title_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Jaccard distance between search_term and product_description... ", end="", flush=True)
    t1 = time.time()
    df["jaccard_search_term_product_description"] = jaccard_distance(df, "search_term_stem", "product_description_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Jaccard distance between search_term and brand... ", end="", flush=True)
    t1 = time.time()
    df["jaccard_search_term_brand"] = jaccard_distance(df, "search_term_stem", "brand_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Jaccard distance between search_term and material... ", end="", flush=True)
    t1 = time.time()
    df["jaccard_search_term_material"] = jaccard_distance(df, "search_term_stem", "material_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Jaccard distance between search_term and color_family... ", end="", flush=True)
    t1 = time.time()
    df["jaccard_search_term_color_family"] = jaccard_distance(df, "search_term_stem", "color_family_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Levenshtein distance between search_term and product_title... ", end="", flush=True)
    t1 = time.time()
    df["levenshtein_search_term_product_title"] = levenshtein_distance(df, "search_term_stem", "product_title_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Levenshtein distance between search_term and product_description... ", end="", flush=True)
    t1 = time.time()
    df["levenshtein_search_term_product_description"] = levenshtein_distance(df, "search_term_stem", "product_description_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Levenshtein distance between search_term and brand... ", end="", flush=True)
    t1 = time.time()
    df["levenshtein_search_term_brand"] = levenshtein_distance(df, "search_term_stem", "brand_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Levenshtein distance between search_term and material... ", end="", flush=True)
    t1 = time.time()
    df["levenshtein_search_term_material"] = levenshtein_distance(df, "search_term_stem", "material_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Levenshtein distance between search_term and color_family... ", end="", flush=True)
    t1 = time.time()
    df["levenshtein_search_term_color_family"] = levenshtein_distance(df, "search_term_stem", "color_family_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Jaro distance between search_term and product_title... ", end="", flush=True)
    t1 = time.time()
    df["jaro_search_term_product_title"] = jaro_distance(df, "search_term_stem", "product_title_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Jaro distance between search_term and product_description... ", end="", flush=True)
    t1 = time.time()
    df["jaro_search_term_product_description"] = jaro_distance(df, "search_term_stem", "product_description_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Jaro distance between search_term and brand... ", end="", flush=True)
    t1 = time.time()
    df["jaro_search_term_brand"] = jaro_distance(df, "search_term_stem", "brand_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Jaro distance between search_term and material... ", end="", flush=True)
    t1 = time.time()
    df["jaro_search_term_material"] = jaro_distance(df, "search_term_stem", "material_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Calculating Jaro distance between search_term and color_family... ", end="", flush=True)
    t1 = time.time()
    df["jaro_search_term_color_family"] = jaro_distance(df, "search_term_stem", "color_family_stem")
    print("Done (%d secs)" % (time.time() - t1))

    print("Adding number of occurrences by product_uid feature... ", end="", flush=True)
    t1 = time.time()
    # count() returns a Series in this case
    count_by_product_uid = df.groupby("product_uid").product_uid.count()
    df["count"] = df["product_uid"].map(lambda x: count_by_product_uid[x])
    print("Done (%d secs)" % (time.time() - t1))

    # print("Caching debug file to %s" % "./cache/train_debug.csv")
    # df.to_csv("./cache/train_debug.csv", encoding="UTF-8", index=False)
    # print("Done (%d secs)" % (time.time() - t1))

    print("Removing unused features... ", end="", flush=True)
    t1 = time.time()
    df.drop("product_uid", axis=1, inplace=True)
    df.drop("search_term", axis=1, inplace=True)
    df.drop("product_title", axis=1, inplace=True)
    df.drop("product_description", axis=1, inplace=True)
    df.drop("brand", axis=1, inplace=True)
    df.drop("material", axis=1, inplace=True)
    df.drop("color_family", axis=1, inplace=True)
    df.drop("search_term_token", axis=1, inplace=True)
    df.drop("product_title_token", axis=1, inplace=True)
    df.drop("product_description_token", axis=1, inplace=True)
    df.drop("brand_token", axis=1, inplace=True)
    df.drop("material_token", axis=1, inplace=True)
    df.drop("color_family_token", axis=1, inplace=True)
    df.drop("search_term_stem", axis=1, inplace=True)
    df.drop("product_title_stem", axis=1, inplace=True)
    df.drop("product_description_stem", axis=1, inplace=True)
    df.drop("brand_stem", axis=1, inplace=True)
    df.drop("material_stem", axis=1, inplace=True)
    df.drop("color_family_stem", axis=1, inplace=True)
    print("Done (%d secs)" % (time.time() - t1))

    return df


def train_model(df, dump):

    ts_main = time.time()

    model = XGBRegressor(
        seed=SEED,
        silent=True,
        objective="reg:linear",
        n_estimators=1000,
        learning_rate=0.1
    )

#    ## Best param:
#    {'max_depth': 7, 'n_estimators': 750, 'colsample_bytree': 0.6, 'learning_rate': 0.01, 'reg_alpha': 0.7,
#     'subsample': 0.9, 'colsample_bylevel': 1.0, 'min_child_weight': 0.5, 'reg_lambda': 0.9}
#    param_grid = {
#        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8],
#        "subsample": [0.80, 0.90, 1.00],
#        "colsample_bytree": [0.60],
#        "min_child_weight": [0.50, 0.75, 1.00],
#        "colsample_bylevel": [1.00],
#        "reg_alpha": [0.60, 0.70, 0.80, 0.90, 1.00],
#        "reg_lambda": [0.60, 0.70, 0.80, 0.90, 1.00],
#        "n_estimators": [100, 300, 500, 750, 1000, 1250, 1500],
#        "learning_rate": [0.1, 0.05, 0.01]
#        }

    param_grid = {
        "max_depth": [7],
        "subsample": [0.90],
        "colsample_bytree": [0.60],
        "colsample_bylevel": [1.00],
        "reg_alpha": [0.70],
        "reg_lambda": [0.90],
        "n_estimators": [1750],
        "learning_rate": [0.01]
        }

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    grid = RandomizedSearchCV(
        estimator=model, param_distributions=param_grid,
        verbose=3, n_jobs=-1, cv=CV_N_FOLDS, scoring=rmse_scorer, n_iter=1,
        random_state=SEED)

    grid.fit(
        df.drop(["relevance", "id"], axis=1).values,
        df["relevance"].values)

    print("## Grid scores: ")
    print(grid.grid_scores_)
    print("## Best score: ")
    print(grid.best_score_)
    print("## Best param: ")
    print(grid.best_params_)
    print("## Best estimator: ")
    print(grid.best_estimator_)

    print("Dumping model at %s" % dump)
    joblib.dump(grid.best_estimator_, dump)

    print("Total time %d secs" % (time.time() - ts_main))
    return grid.best_estimator_


def load_or_train(df):
    """Return a trained model.

    If a cached instance is available returns that one, otherwise a new model
    is trained.
    """

    try:
        model = joblib.load(MODEL_CACHE)
        print("Model loaded from %s" % MODEL_CACHE)
    except OSError:
        model = train_model(df, MODEL_CACHE)

#    print("Confusion matrix:")
#    print(confusion_matrix(df.relevance, model.predict(df.drop(["relevance", "id"], axis=1).values)))

    print("Caching debug file to %s" % "./cache/train_debug.csv")
    t1 = time.time()
    df_with_predictions = df
    df_with_predictions["predictions"] = model.predict(df.drop(["relevance", "id"], axis=1).values)
    df_with_predictions.to_csv("./cache/train_debug.csv", encoding="UTF-8", index=False)
    print("Done (%d secs)" % (time.time() - t1))

    return model


def main():
    df_train = pd.read_csv(TRAIN_CSV, header=0, sep=",", encoding="ISO-8859-1")
    df_test = pd.read_csv(TEST_CSV, header=0, sep=",", encoding="ISO-8859-1")
    df_attributes = pd.read_csv(ATTRIBUTES_CSV, header=0, sep=",", encoding="UTF-8")
    df_descriptions = pd.read_csv(DESCRIPTIONS_CSV, header=0, sep=",", encoding="UTF-8")

    df_train = merge_dataframe(df_train, df_descriptions, df_attributes)
    df_test = merge_dataframe(df_test, df_descriptions, df_attributes)

    df_train = clean_data(df_train)
    df_test = clean_data(df_test)

    df_train = tokenize_data(df_train)
    df_test = tokenize_data(df_test)

    df_train = stem_data(df_train)
    df_test = stem_data(df_test)

    tfidf = build_tfidf(df_train, df_test)

    df_train = feature_engineering(df_train, tfidf)

    model = load_or_train(df_train)

    feature_importances = dict()
    tot = 0
    for i, v in enumerate(model._Booster.get_fscore()):
        n = int(v.strip('f'))
        tot += n
        feature_importances[df_train.drop(["relevance", "id"], axis=1).columns[i]] = n

    print("Top 10 features by importance:")
    for feature in sorted(feature_importances, key=feature_importances.get, reverse=True)[:10]:
        if feature_importances[feature] >= 1.0 / len(df_train.drop(["relevance", "id"], axis=1).columns):
            print(" * %s %0.3f %%" % (feature, feature_importances[feature] / tot))

    df_test = feature_engineering(df_test, tfidf)

    predictions = model.predict(df_test.drop(["id"], axis=1).values)

    df_test["relevance"] = pd.Series(predictions)
    df_test["relevance"] = df_test.relevance.map(lambda x: 3 if x > 3 else x)
    df_test["relevance"] = df_test.relevance.map(lambda x: 1 if x < 1 else x)

    print("Saving predictions to file %s" % PREDS_CACHE)
    df_test.to_csv(PREDS_CACHE, columns=["id", "relevance"], index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
