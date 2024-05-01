token="<hugging_face_api_token>"

from langchain  import HuggingFaceHub
from langchain.chains import LLMChain, LLMRequestsChain
from langchain import PromptTemplate
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token


template = """Between >>> and <<< are the raw search result text from google.
Extract the answer to the question '{query}' or say "not found" if the information is not contained. Provide precise answer only. 
Look for exact values requested and answer accordingly. 
Use the format
Extracted:<answer or "not found">
>>> {requests_result} <<<
Extracted:"""
model=HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct")
model.model_kwargs = {"device_map": "auto","max_length":20,"max_tokens":20}
PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=template,
)

chain = LLMRequestsChain(llm_chain=LLMChain(llm=model, prompt=PROMPT))
question = "What are the top three biggest countries, and their respective sizes?"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+"),
}
print(chain(inputs))
