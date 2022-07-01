import glob
from sklearn.metrics import f1_score
import sys
from datasets import Dataset
import torch
from tqdm import tqdm
from sklearn import metrics
import argparse
from pysentimiento import preprocessing
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import RobertaConfig, RobertaModel
from transformers import RobertaTokenizer

parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument('components', type=str, nargs='+', help="Name of the component that wants to be identified")
parser.add_argument('--sover', type=str, default="liblinear", help="Name of the solver to be used in the Logistic regression")
parser.add_argument('--c', type=float, default=1.0, help="Inverse of regularization strength")

args = parser.parse_args()

REP = 0
NUMBER_OF_PARTITIONS = 10
C = args.c
SOLVER = args.solver
components = args.components
component = components[0]


def labelComponents(text, component_text):
    if len(text.strip()) == 0:
        return []
    if len(component_text) == 0:
        return [0] * len(text.strip().split())

    if component_text[0] != "" and component_text[0] in text:
        parts = text.split(component_text[0])
        rec1 = labelComponents(parts[0], component_text[1:])
        rec2 = []
        if len(parts) > 2:
            rec2 = labelComponents(component_text[0].join(parts[1:]), component_text)
        else:
            rec2 = labelComponents(parts[1], component_text[1:])
        return rec1 + [1] * len(component_text[0].strip().split()) + rec2
    return [0] * len(text.strip().split())


def labelComponentsFromAllExamples(filePatterns, component):
    all_tweets = []
    all_labels = []
#    if multidataset:
#        datasets = []
    for filePattern in filePatterns:
        for f in glob.glob(filePattern):
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
             # TODO: sacar todos los caracteres especiales
            tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "")
            component_text = []
            if component == "Collective":
                property_text = []
            is_argumentative = True
            filesize = 0
            for idx, word in enumerate(annotations):
                filesize += 1
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].lstrip()
                    if current_component.startswith("NonArgumentative"):
                        is_argumentative = False
                        break
                    if current_component.startswith(component):
                        component_text.append(ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", ""))
                    if component == "Collective" and current_component.startswith("Property"):
                        property_text.append(ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", ""))


            if component == "Collective":
                tweet_text += " Property: " + " ".join(property_text)
            preprocessed_text = preprocessing.preprocess_tweet(tweet_text) 
            normalized_text = normalize_text(preprocessed_text, component_text)
            labels = labelComponents(" ".join(normalized_text), component_text)
            if not is_argumentative or filesize == 0:
                continue

#            elif multidataset:
#                dicc = {"tokens": [normalized_text], "labels": [labels]}
#                datasets.append([Dataset.from_dict(dicc), normalized_text])
            else:
                all_tweets.append(normalized_text)
                all_labels.append(labels)

#    if multidataset:
#        return datasets

#    ans = {"tokens": all_tweets, "labels": all_labels}
    all_tweets_flattened = [word for tweet in all_tweets for word in tweet]
    all_labels_flattened = [label for labels in all_labels for label in labels]
    return pd.DataFrame([all_tweets_flattened, all_labels_flattened], index=["text", "labels"]).T

#def tokenize_and_align_labels(dataset, tokenizer, is_multi = False):
#    def tokenize_and_align_labels_per_example(example):
#        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
#
#        labels = []
#        for i, label in enumerate(example[f"labels"]):
#            word_ids = tokenized_inputs.word_ids(batch_index=i)
#            previous_word_idx = None
#            label_ids = []
#            for word_idx in word_ids:  # Set the special tokens to -100.
#                if word_idx is None:
#                    label_ids.append(-100)
#                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                    label_ids.append(label[word_idx])
#                else:
#                    label_ids.append(-100)
#                previous_word_idx = word_idx
#            labels.append(label_ids)
#
#        tokenized_inputs["labels"] = labels
#        return tokenized_inputs
#
#    if is_multi:
#        return [{"dataset": data[0].map(tokenize_and_align_labels_per_example, batched=True), "text": data[1]} for data in dataset]
#    return dataset.map(tokenize_and_align_labels_per_example, batched=True)

def normalize_text(tweet_text, arg_components_text):
    parts_processed = []
    splitted_text = [tweet_text]
    for splitter in arg_components_text:
        for segment in splitted_text:
            if segment != '':
                new_splitted_text = []
                new_split = segment.split(splitter)
                for idx, splitt in enumerate(new_split):
                    new_splitted_text.append(splitt)
        splitted_text = new_splitted_text

    reconstructed_text = []
    current_text = tweet_text
    for part in splitted_text:
        if (part != ''):
            spp = current_text.split(part)
            for word in spp[0].split():
                reconstructed_text.append(word)
            for word in part.split():
                reconstructed_text.append(word)
            current_text = part.join(spp[1:])
    return reconstructed_text
    


    for idx, part in enumerate(splitted_text):
        for word in part.strip().split():
            parts_processed.append(word)

    return parts_processed


def tokenize_examples(dataset, tokenizer):
    def tokenize_and_align_labels_per_example(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(example["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    dataset["text_tokenized"] = dataset.apply(tokenize_and_align_labels_per_example, axis=1)
    return dataset

def train(model, embeddings_model, tokenizer, train_partition_patterns, dev_partition_patterns, test_partition_patterns, component, with_embeddings = False):

    

    training_set = labelComponentsFromAllExamples(train_partition_patterns, component)
    if with_embeddings:
        training_set = tokenize_examples(training_set, tokenizer)
    
    X = training_set.drop("labels", axis=1)
    X.fillna(0, inplace=True)


    
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(X.to_dict('records'))
    y = training_set.labels.values
    y = y.astype('int')


    classes = np.unique(y)
#    print(classes.tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle=False)

    logreg = LogisticRegression(C=c, solver=solver, class_weight = "balanced")
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    filename = "results_test_{}_{}_LR_{}_{}".format(C, SOLVER, REP, component)

    with open(filename, 'w') as w:
        w.write("{},{},{},{}".format(accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average="binary", pos_label='1'), recall_score(y_test, y_pred, average="binary", pos_label='1'), f1_score(y_test, y_pred, average="binary", pos_label='1'))

#    print(y_pred)

#    print(f1_score(y_test, y_pred))


#    print(list(zip(y_test, y_pred)))

#    print(X)

filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, NUMBER_OF_PARTITIONS + 1)]
dataset_combinations = [[filePatterns[:9], filePatterns[9:]], [filePatterns[1:], filePatterns[0:1]], [[*filePatterns[:1], *filePatterns[2:]], filePatterns[1:2]]]

for combination in dataset_combinations:
    REP = REP + 1
    for cmpnent in components:
        component = cmpnent
        model = LogisticRegression()
        configuration = RobertaConfig()
        embeddings_model = RobertaModel(configuration)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        train(model, embeddings_model, tokenizer, filePatterns, combination[0], combination[1], cmpnent)

