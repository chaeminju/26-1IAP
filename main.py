from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime

app = FastAPI()

# 데이터 형식 정의
class DetectionLog(BaseModel):
    device_id : str
    timestamp : datetime
    loss_value : float = Field(..., ge=0)
    risk_level : int = Field(..., ge=0, le=10)  # 위험도는 0에서 10 사이의 정수

@app.get("/")
def root():
    return {"message" : "확인중"}

@app.post("/log/detection")
def receive_log(log: DetectionLog):
    print(f"Received Log: {log.device_id}, 위험도: {log.risk_level}, loss={log.loss_value}")
    return {"status": "log received", "data": log.model_dump()}

