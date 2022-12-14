import glob
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import torch
from transformers import EvalPrediction
from sklearn import metrics
import argparse
from transformers import EarlyStoppingCallback
import random
from typing import Optional
import torch.nn.functional as F


parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument('components', type=str, nargs='+', help="Name of the component that wants to be identified")
parser.add_argument('--modelname', type=str, default="roberta-base", help="Name of the language model to be downloaded from huggingface")
parser.add_argument('--lr', type=float, default=2e-05, help="Learning rate for training the model. Default value is 2e-05")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation. Default size is 16")
parser.add_argument('--add_annotator_info', type=bool, default=False, help="For Pivot and Collective add information about premises and Property respectively that an annotator would have when annotating these components")
parser.add_argument('--type_of_premise', type=bool, default=False, help="If true, model will be trained to predict the type of premises. If true, only valid components are Justification and Conclusion")
parser.add_argument('--simultaneous_components', type=bool, default=False, help="Set to true if trying to do joint predictions. Possible values for components are Collective-Property or Premises")
parser.add_argument('--all_components', type=bool, default=False, help="Set to true if trying to do multilabel classification predicting all components for each word simultaneously. Component value must be set to 'All'")

args = parser.parse_args()


LEARNING_RATE = args.lr
NUMBER_OF_PARTITIONS = 10
device = torch.device("cuda:0")
BATCH_SIZE = args.batch_size
EPOCHS = 20 * (BATCH_SIZE / 16)
MODEL_NAME = args.modelname
REP=0
FOLDS=3
components = args.components
component = components[0]
add_annotator_info = args.add_annotator_info
type_of_premise = args.type_of_premise
simultaneous_components = args.simultaneous_components
all_components = args.all_components
quadrant_types_to_label = {"fact": 0, "value": 1, "policy": 2}


class MultiLabelTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.FloatTensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)
            # logging.info(f"Using multi-label classification with class weights", class_weights)
        self.loss_fct = torch.nn.BCEWithLogitsLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        outputs = F.sigmoid(outputs.logits)
        try:
            loss = self.loss_fct(outputs.view(-1), labels.view(-1))
        except AttributeError:  # DataParallel
            loss = self.loss_fct(outputs.logits.view(-1, model.module.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_metrics_f1(p: EvalPrediction):

    if all_components:
        preds = p.predictions
        labels = p.label_ids

        print("========================================================================")
        print("EVALUATION")


        true_labels = [[str(l[0]) for l in label if l[0] != -100] for label in labels]
        true_predictions = [
            [str(round(p[0])) for (p, l) in zip(prediction, label) if l[0] != -100]
            for prediction, label in zip(preds, labels)
        ]
        print(true_labels)
        print(true_predictions)
        print(len(true_labels))
        print(len(true_predictions))
        print(len(preds))
        print(len(labels))
        all_true_labels = [l for label in true_labels for l in label]
        all_true_preds = [p for preed in true_predictions for p in preed]
        if simultaneous_components:
            avrge = "macro"
            f1_all = metrics.f1_score(all_true_labels, all_true_preds, average=None)
        else:
            avrge = "binary"    

    else:
        preds = p.predictions.argmax(-1)
        labels = p.label_ids

        print("========================================================================")
        print("EVALUATION")
        print(preds)
        print(labels)

        if not type_of_premise and component != "Argumentative":
            true_labels = [[str(l) for l in label if l != -100] for label in labels]
            true_predictions = [
                [str(p) for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(preds, labels)
            ]
            all_true_labels = [l for label in true_labels for l in label]
            all_true_preds = [p for preed in true_predictions for p in preed]
            if simultaneous_components:
                avrge = "macro"
                f1_all = metrics.f1_score(all_true_labels, all_true_preds, average=None)
            else:
                avrge = "binary"
        else:
            all_true_labels = [str(label) for label in labels]
            all_true_preds = [str(pred) for pred in preds]
            if type_of_premise:
                avrge = "macro"
                f1_all = metrics.f1_score(all_true_labels, all_true_preds, average=None)
            else:
                avrge = "binary"


    f1 = metrics.f1_score(all_true_labels, all_true_preds, average=avrge, pos_label='1')

    acc = metrics.accuracy_score(all_true_labels, all_true_preds)

    recall = metrics.recall_score(all_true_labels, all_true_preds, average=avrge, pos_label='1')

    precision = metrics.precision_score(all_true_labels, all_true_preds, average=avrge, pos_label='1')

    f1_micro = metrics.f1_score(all_true_labels, all_true_preds, average="micro")

    recall_micro = metrics.recall_score(all_true_labels, all_true_preds, average="micro")

    precision_micro = metrics.precision_score(all_true_labels, all_true_preds, average="micro")

    confusion_matrix = metrics.confusion_matrix(all_true_labels, all_true_preds)


    w = open("./results_{}_{}_{}_{}_{}-metrics".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), BATCH_SIZE, component, REP), "a")

    w.write("{},{},{},{},{},{},{}\n".format(str(acc), str(f1), str(precision), str(recall), str(f1_micro), str(precision_micro), str(recall_micro)))
    w.close()

    ans = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': str(confusion_matrix),
    }

    if type_of_premise or simultaneous_components:
        ans['f1_all'] = str(f1_all)

    return ans



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
    return [0.0] * len(text.strip().split())

def delete_unwanted_chars(text):
    return text.replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", "").replace("·", "")

def getLabel(label):
    if label == "O":
        return 0.0
    else:
        return 1.0


def labelComponentsFromAllExamples(filePatterns, component, multidataset = False, add_annotator_info = False, isTypeOfPremise = False, multiple_components = False, all_components=False):
    all_tweets = []
    all_labels = []
    if multidataset:
        datasets = []
    for filePattern in filePatterns:
        for f in glob.glob(filePattern):
            conll_file = open(f, 'r')
            tweet = []
            if not isTypeOfPremise:
                labels = []
            else:
                labels = -1
            if add_annotator_info:
                if component == "Collective":
                    property_text = []
                if component == "pivot":
                    justification_text = []
                    conclusion_text = []
            is_argumentative = True
            for idx, line in enumerate(conll_file):
                line_splitted = line.split("\t")
                word = line_splitted[0]
                if not isTypeOfPremise:
                    tweet.append(word)
                else:
                    if component == "Premise2Justification":
                        if line_splitted[2] != "O":
                            tweet.append(word)
                    elif component == "Premise1Conclusion":
                        if line_splitted[3] != "O":
                            tweet.append(word)
                if line_splitted[1] != "O" or not is_argumentative:
                    is_argumentative = False
                    continue


                if isTypeOfPremise:
                    if component == "Premise2Justification":
                        if line_splitted[2] != "O":
                            labels = quadrant_types_to_label[line_splitted[7].replace("\n", "")]
                    elif component == "Premise1Conclusion":
                        if line_splitted[3] != "O":
                            labels = quadrant_types_to_label[line_splitted[8].replace("\n", "")]
                elif multiple_components:
                    if component == "Collective-Property":
                        col = getLabel(line_splitted[4])
                        prop = getLabel(line_splitted[5]) * 2
                        labels.append(max(col, prop))
                    if component == "Premises":
                        just = getLabel(line_splitted[2])
                        conc = getLabel(line_splitted[3]) * 2
                        labels.append(max(just, conc))
                
                elif all_components:
                    label = [getLabel(v) for v in line_splitted[1:7]]
                    labels.append(label)
                else:
                    if component == "Premise2Justification":
                        labels.append(getLabel(line_splitted[2]))
                    elif component == "Premise1Conclusion":
                        labels.append(getLabel(line_splitted[3]))
                    elif component == "Collective":
                        labels.append(getLabel(line_splitted[4]))
                        if add_annotator_info and getLabel(line_splitted[5]) == 1:
                            property_text.append(word)
                    elif component == "Property":
                        labels.append(getLabel(line_splitted[5]))
                    elif component == "pivot":
                        labels.append(getLabel(line_splitted[6]))
                        if add_annotator_info and getLabel(line_splitted[2]) == 1:
                            justification_text.append(word)
                        if add_annotator_info and getLabel(line_splitted[3]) == 1:
                            conclusion_text.append(word)

            if component == "Argumentative":
                labels = 1.0 if is_argumentative else 0.0
            if not is_argumentative and component != "Argumentative":
                if all_components:
                    labels = [0.0] * 6
                else:
                    continue
            if isTypeOfPremise:
                assert(labels >= 0)
            if add_annotator_info:
                to_add = []
                if component == "Collective":
                    to_add = ["Property:"] + property_text
                if component == "pivot":
                    to_add = ["Justification:"] + justification_text + ["Conclusion:"] + conclusion_text
                tweet += to_add
                labels += [0.0] * len(to_add)

            if multidataset:
                dicc = {"tokens": [tweet], "labels": [labels]}
                datasets.append([Dataset.from_dict(dicc), tweet])
            else:
                all_tweets.append(tweet)
                all_labels.append(labels)


    if multidataset:
        return datasets

    ans = {"tokens": all_tweets, "labels": all_labels}
    return Dataset.from_dict(ans)


def tokenize_and_align_labels(dataset, tokenizer, is_multi = False, is_bertweet=False, one_label_per_example=False, all_components=False):
    def tokenize_and_align_labels_one_label(example):
        return tokenizer(example["tokens"], truncation=True, is_split_into_words=True)

    def tokenize_and_align_labels_per_example(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(example[f"labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100.0)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100.0)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_and_align_labels_per_example_multilabel(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, padding=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(example[f"labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append([-100.0]*6)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append([-100.0]*6)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_and_align_labels_per_example_bertweet(example):
        tkns = example["tokens"]
        labels = example["labels"]
        if len(tkns) == 0 and len(labels) == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}
        tokenized_input = tokenizer(tkns, truncation=True, is_split_into_words=True)
        label_ids = [-100]
        for word, label in zip(tkns, labels):
            tokens = tokenizer(word).input_ids
            label_ids.append(label)
            for i in range(len(tokens)-3):
                label_ids.append(-100.0)
        label_ids.append(-100)
        assert(len(tokenized_input.input_ids) == len(label_ids))
        assert(len(tokenized_input.input_ids) == len(tokenized_input.attention_mask))
        return {"input_ids": tokenized_input.input_ids, "labels": label_ids, "attention_mask": tokenized_input.attention_mask}

    if one_label_per_example:
        function_to_apply = tokenize_and_align_labels_one_label
    elif all_components:
        function_to_apply = tokenize_and_align_labels_per_example_multilabel
    else:
        function_to_apply = tokenize_and_align_labels_per_example
        if is_bertweet:
            function_to_apply = tokenize_and_align_labels_per_example_bertweet
            if is_multi:
                return [{"dataset": data[0].map(function_to_apply), "text": data[1]} for data in dataset]
            return dataset.map(function_to_apply)
    if is_multi:
        return [{"dataset": data[0].map(function_to_apply, batched=True), "text": data[1]} for data in dataset]
    return dataset.map(function_to_apply, batched=True)



def train(model, tokenizer, train_partition_patterns, dev_partition_patterns, test_partition_patterns, component, is_bertweet=False, add_annotator_info=False, is_type_of_premise=False, multiple_components = False, all_components=False):

    training_set = tokenize_and_align_labels(labelComponentsFromAllExamples(train_partition_patterns, component, add_annotator_info=add_annotator_info, isTypeOfPremise=is_type_of_premise, multiple_components=multiple_components, all_components=all_components), tokenizer, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative"), all_components=all_components)
    dev_set = tokenize_and_align_labels(labelComponentsFromAllExamples(dev_partition_patterns, component, add_annotator_info=add_annotator_info, isTypeOfPremise=is_type_of_premise, multiple_components=multiple_components, all_components=all_components), tokenizer, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative"), all_components=all_components)
    test_set = tokenize_and_align_labels(labelComponentsFromAllExamples(test_partition_patterns, component, add_annotator_info=add_annotator_info, isTypeOfPremise=is_type_of_premise, multiple_components=multiple_components, all_components=all_components), tokenizer, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative"), all_components=all_components)
    test_set_one_example = tokenize_and_align_labels(labelComponentsFromAllExamples(test_partition_patterns, component, multidataset = True, add_annotator_info=add_annotator_info, isTypeOfPremise=is_type_of_premise, multiple_components=multiple_components, all_components=all_components), tokenizer, is_multi = True, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative"), all_components=all_components)
    
    training_args = TrainingArguments(
        output_dir="./results_eval_{}_{}".format(MODEL_NAME.replace("/", "-"), component),
        evaluation_strategy="steps",
        eval_steps=10,
        save_total_limit=8,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.05,
        report_to="none",
        metric_for_best_model='f1',
        load_best_model_at_end=True
    )

    if not all_components:

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_set,
            eval_dataset=dev_set,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics= compute_metrics_f1,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=4)]
        )
    
    else:
        trainer = MultiLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=training_set,
            eval_dataset=dev_set,
            tokenizer=tokenizer,
            # data_collator=data_collator,
            compute_metrics= compute_metrics_f1,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=4)]
        )

    trainer.train()

    results = trainer.predict(test_set)
    if not type_of_premise:
        filename = "./results_test_{}_{}_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), BATCH_SIZE, REP, component)
    else:
        filename = "./results_test_{}_{}_{}_{}_{}_type-of-premise".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), BATCH_SIZE, REP, component)
    with open(filename, "w") as writer:
        if type_of_premise or multiple_components:
            writer.write("{},{},{},{},{}\n".format(results.metrics["test_accuracy"], results.metrics["test_f1"], results.metrics["test_precision"], results.metrics["test_recall"], results.metrics["test_f1_all"]))
        else:
            writer.write("{},{},{},{}\n".format(results.metrics["test_accuracy"], results.metrics["test_f1"], results.metrics["test_precision"], results.metrics["test_recall"]))
        writer.write("{}".format(str(results.metrics["test_confusion_matrix"])))

    # examples_filename = "./examples_test_{}_{}_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), BATCH_SIZE, REP, component)
    # with open(examples_filename, "w") as writer:
    #     for dtset in test_set_one_example:
    #         result = trainer.predict(dtset["dataset"])
    #         preds = result.predictions.argmax(-1)[0]
    #         if component == "Argumentative":
    #             preds = [preds]
    #         assert (len(preds) == len(result.label_ids))
    #         comparison = [(truth, pred) for truth, pred in zip(result.label_ids, preds) if truth != -100]
    #         writer.write("Tweet:\n")
    #         writer.write("{}\n".format(dtset["dataset"]["tokens"][0]))
    #         if component == "Argumentative":
    #             writer.write("{} - {}".format(comparison[0][0], comparison[0][1]))
    #         else:
    #             for word, pair in zip(dtset["dataset"]["tokens"][0], comparison):
    #                 writer.write("{}\t\t\t{}\t{}\n".format(word, pair[0], pair[1]))
    #         writer.write("-------------------------------------------------------------------------------\n")




filePatterns = ["./datasets_CoNLL/english/hate_tweet_*.conll"]

allFiles = []
for pattern in filePatterns:
    for f in glob.glob(pattern):
        allFiles.append(f)

dataset_combinations = []
for i in range(FOLDS):
    allFilesCp = allFiles.copy()
    random.Random(41 + i).shuffle(allFilesCp)
    dataset_combinations.append([allFilesCp[:770], allFilesCp[770:870], allFilesCp[870:]])

for combination in dataset_combinations:
    REP = REP + 1
    for cmpnent in components:
        component = cmpnent
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
        if not type_of_premise and component != "Argumentative":
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
            if simultaneous_components:
                output_num = 3
            elif all_components:
                output_num = 6
            else:
                output_num = 2
            model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=output_num)
            
        else:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            if type_of_premise:
                 output_num = 3
            else:
                 output_num = 2
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=output_num)

        model.to(device)
        train(model, tokenizer, combination[0], combination[1], combination[2], cmpnent, is_bertweet = MODEL_NAME == "bertweet-base", add_annotator_info=add_annotator_info, is_type_of_premise = type_of_premise, multiple_components=simultaneous_components, all_components=all_components)


