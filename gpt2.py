import inseq
import numpy as np
import evaluate
import sklearn
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer
from datasets import load_dataset_builder, load_dataset, load_metric, Dataset
import pandas as pd

    # function to be used for inspecting the dataset, without downloading it
def inspect_dataset(name, subset):
    ds_builder = load_dataset_builder(name, subset)
    print(ds_builder.info.description)

    # Construct a dataset for the experiment by stripping the queries of their last words
    # Then, we store the different words of the "sentence_good" and "sentence_bad" categories in
    # their own fields and create a dataset using those three lists as columns.
def construct_dataset_for_experiment(name, subset, data_split = "train"):
    dataset = load_dataset(name, subset, split=data_split)

    sentences_for_classification = []
    words_good = []
    for sentence in dataset["sentence_good"]:
        split_sentence = sentence.rsplit(" ", 1)  # split sentence from the right, seperating the last word

        correct_word = split_sentence[1][:-1]  # the last word is the right answer, minus the '.'
        words_good.append(correct_word)
        sentences_for_classification.append(split_sentence[0])  # the rest of the sentence is the query

    words_bad = []
    for sentence in dataset["sentence_bad"]:
        split_sentence = sentence.rsplit(" ", 1)  # split sentence from the right, seperating the last word
        incorrect_word = split_sentence[1][:-1]  # the last word is the wrong answer, minus the '.'
        words_bad.append(incorrect_word)

    zipped = list(zip(sentences_for_classification, words_good, words_bad))
    df = pd.DataFrame(zipped, columns=['query', 'correct', 'incorrect'])
    new_dataset = Dataset.from_pandas(df)
    return new_dataset

    # load a dataset with subset and return preprocessed version
def preprocess(model, dataset):

    tokenizer = AutoTokenizer.from_pretrained(model)

    def tokenization(example):
        # set padding token, will be ignored by attention mask but is necessary for some functionalities of the object
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(example["query"], return_tensors='pt', padding="max_length", truncation = True)

    tokenized = dataset.map(tokenization, batched=True, batch_size = 10)

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # mess_around_with_dataset(dataset)
    return tokenized

def train(model, training_dataset, validation_dataset):
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)
    training_args = TrainingArguments(output_dir = "gpt2testtrainer", evaluation_strategy = "epoch")
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis = 1)
        return metric.compute(predictions, references = labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        )

    trainer.train()

# copied most of this from inseq presentation slides sent to me in email
def test(model, dataset):
    attribution_model = inseq.load_model(model, "input_x_gradient")

    for index, value in enumerate(dataset["query"]):
        # Pre-compute ids and attention map for the contrastive target
        contrast = attribution_model.encode(dataset["query"][index] + " " + dataset["incorrect"][index])

        # Perform the contrastive attribution
        out = attribution_model.attribute(
            dataset["query"][index],
            dataset["query"][index]  + " " + dataset["correct"][index],
            attributed_fn = "contrast_prob_diff",
            contrast_ids = contrast.input_ids,
            contrast_attention_mask = contrast.attention_mask,

            # we also visualize the corresponding step score
            step_scores = ["contrast_prob_diff"]
        )
        out.show()


    # me messing around with dataset
def mess_around_with_dataset(dataset):
    print(dataset[0])
    print(dataset[-1])
    for sentence in dataset["query"]:
        print(sentence)

    # just fucking around with gpt
def try_gpt():
    generator = pipeline('text-generation', model='gpt2')

    set_seed(42)

    dict_list = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

    num = 0
    for dict in dict_list:
        num += 1
        print("Sequence number " + str(num) + ": " + dict["generated_text"])

if __name__ == "__main__":
    name = "blimp"
    subset = "determiner_noun_agreement_1"
    model = "gpt2"

    dataset = construct_dataset_for_experiment(name, subset)
    # mess_around_with_dataset(dataset)
    # training_dataset = preprocess(name, model, subset, "train")
    # validation_dataset = preprocess(name, model, subset, "validation")
    # testing_dataset = preprocess(model, dataset)

    # train(model, training_dataset, validation_dataset)
    test(model, dataset)

    #metric = load_metric(name, subset)
    #final_score = metric.compute(predictions=predictions, references=gold_references)
