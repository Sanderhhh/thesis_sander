from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2Model
from datasets import load_dataset_builder, load_dataset

    # function to be used for inspecting the dataset, without downloading it
def inspect_dataset(name, subset):
    ds_builder = load_dataset_builder(name, subset)
    print(ds_builder.info.description)
    # print(ds_builder.info.features)

    # load a dataset with subset and return preprocessed version
def preprocess(name, subset = None):
    dataset = load_dataset(name, subset, split = "train")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(tokenizer(dataset[0]['text']))

    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    return dataset

    # me messing around with dataset
def mess_around_with_dataset(dataset):
    print(dataset[0])
    print(dataset[-1])
    for sentence in dataset["sentence_good"]:
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
    name = "rotten_tomatoes"
    subset = None
    dataset = preprocess(name, subset)
    print(dataset.format["type"])
