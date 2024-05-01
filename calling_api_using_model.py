from model import tokenizer, foundational_model

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
 
from langchain.chains import APIChain
from langchain.chains.api import open_meteo_docs
import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

template = """Between >>> and <<< are the raw search result text from google.
Extract the answer to the question '{query}' or say "not found" if the information is not contained. Provide precise answer only. 
Look for exact values requested and answer accordingly. 
Use the format
Extracted:<answer or "not found">
>>> {requests_result} <<<
Extracted:"""
 
PROMPT = PromptTemplate.from_template(template=template)
pipe=pipeline("text-generation",model=foundational_model, tokenizer=tokenizer,  max_new_tokens=50)
hf=HuggingFacePipeline(pipeline=pipe)
chain = PROMPT|hf 
question = "What are the top three biggest countries, and their respective sizes"
inputs = {
    "query": question,
    "requests_result": "https://www.google.com/search?q=" + question.replace(" ", "+"),
}
print(chain.invoke(inputs))
chain=APIChain.from_llm_and_api_docs(
    foundational_model, 
    verbose=True,
    limit_to_domains=["https://www.google.com/search?q="],
    
)
print(chain.run(question))




