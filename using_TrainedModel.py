from peft import PeftModel
from model import foundational_model , tokenizer
from util import get_outputs,get_output_directory_prompt , get_output_directory_sentences
import os 
 
output_directory_prompt = get_output_directory_prompt()
output_directory_sentences = get_output_directory_sentences()
loaded_model_prompt = PeftModel.from_pretrained(
    foundational_model,
    output_directory_prompt,
    # device_map='auto',
    is_trainable=False,
)


input_prompt = tokenizer("I want you to act as a motivational coach. ", return_tensors="pt")
loaded_model_prompt_outputs = get_outputs(loaded_model_prompt, input_prompt)
print(tokenizer.batch_decode(loaded_model_prompt_outputs, skip_special_tokens=True))

loaded_model_prompt.load_adapter(output_directory_sentences, adapter_name="quotes")
loaded_model_prompt.set_adapter("quotes")
input_sentences = tokenizer("There are two nice things that should matter to you:", return_tensors="pt")
loaded_model_sentences_outputs = get_outputs(loaded_model_prompt, input_sentences)
print(tokenizer.batch_decode(loaded_model_sentences_outputs, skip_special_tokens=True))