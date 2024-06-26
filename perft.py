# perft.py
from model import foundational_model,tokenizer,model_name,NUM_EPOCHS,NUM_VIRTUAL_TOKENS
import os
from util import get_outputs,get_output_directory_prompt,get_output_directory_sentences
#import pandas as pd
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

from datasets import load_dataset
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit

from transformers import TrainingArguments

from transformers import Trainer, DataCollatorForLanguageModeling


input_prompt = tokenizer("I want you to act as a motivational coach. ", return_tensors="pt")
foundational_outputs_prompt = get_outputs(foundational_model, input_prompt, max_new_tokens=50)

print(tokenizer.batch_decode(foundational_outputs_prompt, skip_special_tokens=True))


input_sentences = tokenizer("There are two nice things that should matter to you:", return_tensors="pt")
foundational_outputs_sentence = get_outputs(foundational_model, input_sentences, max_new_tokens=50)

print(tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True))

dataset_prompt = "fka/awesome-chatgpt-prompts"

# Create the Dataset to create prompts.
data_prompt = load_dataset(dataset_prompt)
data_prompt = data_prompt.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
train_sample_prompt = data_prompt["train"].select(range(50))

print(train_sample_prompt[:1])

dataset_sentences = load_dataset("Abirate/english_quotes")

data_sentences = dataset_sentences.map(lambda samples: tokenizer(samples["quote"]), batched=True)
train_sample_sentences = data_sentences["train"].select(range(25))
train_sample_sentences = train_sample_sentences.remove_columns(["author", "tags"])
print(train_sample_sentences[:1])    



generation_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,  # This type indicates the model will generate text.
    prompt_tuning_init=PromptTuningInit.RANDOM,  # The added virtual tokens are initialized with random numbers
    num_virtual_tokens=NUM_VIRTUAL_TOKENS,  # Number of virtual tokens to be added and trained.
    tokenizer_name_or_path=model_name,  # The pre-trained model.
)

peft_model_prompt = get_peft_model(foundational_model, generation_config)
print(peft_model_prompt.print_trainable_parameters())

peft_model_sentences = get_peft_model(foundational_model, generation_config)
print(peft_model_sentences.print_trainable_parameters())


#creating training arguments
def create_training_arguments(path, learning_rate=0.0035, epochs=NUM_EPOCHS):
    return TrainingArguments(
        output_dir=path,  # The output directory
        use_cpu=True, 
        auto_find_batch_size=True,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        )
    

  
output_directory_prompt=get_output_directory_prompt()
output_directory_sentences=get_output_directory_sentences()

training_args_prompt = create_training_arguments(output_directory_prompt, 0.003, NUM_EPOCHS)
training_args_sentences = create_training_arguments(output_directory_sentences, 0.003, NUM_EPOCHS)

def create_trainer(model, training_args,train_dataset):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        #mlm =False indicate not to use masked language modeling
    )
    
trainer_prompt=create_trainer(peft_model_prompt, training_args_prompt, train_sample_prompt)
trainer_prompt.train()

trainer_sentences=create_trainer(peft_model_prompt, training_args_sentences, train_sample_sentences)
trainer_sentences.train()

trainer_prompt.model.save_pretrained(output_directory_prompt)
trainer_sentences.model.save_pretrained(output_directory_sentences)