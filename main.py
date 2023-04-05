import shutil
import os

from fastapi import FastAPI
from gpt_index import ServiceContext, SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from classes.responseBody import responseBody

load_dotenv()
app = FastAPI()

@app.post("/chat")
async def chatbot(input_text):
    try:
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
        response = index.query(input_text, response_mode="compact")
        return responseBody(message=response.response, success=True)
    except Exception as err:
        return responseBody(message=err.strerror, success=False)
    
@app.get("/resetIndex")
async def construct_index():
    try:
        dateStamp = datetime.now().strftime("%d-%m-%y_%H%M%S")
        dir = os.path.dirname(__file__)
        currentIndex = os.path.join(dir, "index.json")
        archivedIndex = os.path.join(dir, "archive", f"index_{dateStamp}.json")
        shutil.move(currentIndex, archivedIndex)
        
        max_input_size = os.getenv("max_input_size")
        num_outputs = os.getenv("num_outputs")
        max_chunk_overlap = os.getenv("max_chunk_overlap")
        chunk_size_limit = os.getenv("chunk_size_limit")

        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
        documents = SimpleDirectoryReader(os.getenv("docDir")).load_data()
        
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        index.save_to_disk('index.json')

        return responseBody(message=f"Success. Previous index archived at {archivedIndex}", success=True)
    except Exception as err:
        return responseBody(message=err.strerror, success=False)