import uvicorn
import numpy as np
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
import pickle

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


rgModel= pickle.load(open("model.pkl", "rb"))


@app.get('/')
def index():
    return {'message': 'APP DEV TEST'}


@app.get("/predictLoan")
def gePredictLoan(Gender: int,Married: int,Dependents: int,Education:int,Self_Employed: int,ApplicantIncome: int,CoapplicantIncome: int,Loan_Amount_Term : int,Credit_History: int,Property_Area:int ):
    prediction=rgModel.predict([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,Loan_Amount_Term,Credit_History,Property_Area]])
    return{'Loan':prediction[0]}





if __name__ == '__main__':
     uvicorn.run(app, port=80, host='0.0.0.0')
    
    