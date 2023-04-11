import shutil
import os

from fastapi import FastAPI
from gpt_index import ServiceContext, SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from starlette.responses import RedirectResponse
from classes.responseBodies import ResponseBody
from classes.requestBodies import RequestBody, ChatRequest

load_dotenv()
app = FastAPI()

dir = os.path.dirname(__file__)
indexLocation = os.path.join(dir, "index.json")


@app.get("/")
def homepage():
    return RedirectResponse(url='/docs')


@app.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    if request.key != os.getenv("KEY"):
        return ResponseBody(message="Invalid  Access", success=False)
    if not os.path.exists(indexLocation):
        return ResponseBody(message="No data model found, you must use /retrain", success=False)

    try:
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
        response = index.query(request.chatInput, response_mode="compact")
        return ResponseBody(message=response.response, success=True)
    except Exception as err:
        return ResponseBody(message=str(err), success=False)


@app.post("/retrain", tags=["AI Admin"])
async def retrainModel(request: RequestBody):
    if request.key != os.getenv("KEY"):
        return ResponseBody(message="Invalid  Access", success=False)
    try:
        dateStamp = datetime.now().strftime("%d-%m-%y_%H%M%S")
        if os.path.exists(indexLocation):
            archivedIndex = os.path.join(
                dir, "archive", f"index_{dateStamp}.json")
            shutil.move(indexLocation, archivedIndex)

        max_input_size = os.getenv("max_input_size")
        num_outputs = os.getenv("num_outputs")
        max_chunk_overlap = os.getenv("max_chunk_overlap")
        chunk_size_limit = os.getenv("chunk_size_limit")

        prompt_helper = PromptHelper(
            max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
        llm_predictor = LLMPredictor(llm=OpenAI(
            temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
        documents = SimpleDirectoryReader(os.getenv("docDir")).load_data()

        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTSimpleVectorIndex.from_documents(
            documents, service_context=service_context)
        index.save_to_disk('index.json')

        return ResponseBody(message=f"Success.", success=True)
    except Exception as err:
        return ResponseBody(message=str(err), success=False)
