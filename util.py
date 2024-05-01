from model import tokenizer
import os
# this function returns the outputs from the model received, and inputs.
def get_outputs(model, inputs, max_new_tokens=100):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        # temperature=0.2,
        # top_p=0.95,
        # do_sample=True,
        repetition_penalty=1.5,  # Avoid repetition.
        early_stopping=False,  # The model can stop before reach the max_length
        eos_token_id=tokenizer.eos_token_id,
    )
    return outputs
def get_working_dir():
    return "./"


def get_output_directories():
    working_dir = get_working_dir()
    output_directory_prompt = os.path.join(working_dir, "peft_outputs_prompt")
    output_directory_sentences = os.path.join(working_dir, "peft_outputs_sentences")

    # Just creating the directors if not exist.
    for directory in [working_dir, output_directory_prompt, output_directory_sentences]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    return output_directory_prompt, output_directory_sentences


def get_output_directory_prompt():
    return get_output_directories()[0]

def get_output_directory_sentences():
    return get_output_directories()[1]

