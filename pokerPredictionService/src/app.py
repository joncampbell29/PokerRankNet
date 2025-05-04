from fastapi import FastAPI
from api.pokerInterface import router

app = FastAPI()



@app.get("/health")
def health_check(self):
    return {"status": "up"}

app.include_router(router)