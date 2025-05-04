from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()



@app.get("/health")
def health_check(self):
    return {"status": "up"}
