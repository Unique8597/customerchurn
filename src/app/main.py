from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message":"welcome to CustomerChurn API"}

@app.post("/predict")
def predict():
    return {"message":"This is the prediction end point"}