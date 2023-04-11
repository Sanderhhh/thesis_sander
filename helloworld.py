from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')

set_seed(42)

dict_list = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

num = 0
for dict in dict_list:
    num += 1
    print("Sequence number " + str(num) + ": " + dict["generated_text"])
