import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)  
from fastapi import FastAPI,Request,Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model.model import __version__ as model_version
from model.model import predict_pipeline


 
templates = Jinja2Templates(directory='templates')
app = FastAPI()

class textIn(BaseModel):
    text : str

class predictonOut(BaseModel):
    language : str


@app.get('/',response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse('index.html',{'request':request})

@app.post('/',response_model=predictonOut)
async def prediction(request:Request,text:str = Form(...)):
    language = predict_pipeline(text)
    return templates.TemplateResponse('index.html',{'request':request,'language':language})
