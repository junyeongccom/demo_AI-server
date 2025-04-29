from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from app.domain.api.crime_router import router as crime_router

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="Crime Analysis Service",
    description="서울시 범죄 데이터 분석 API",
    version="0.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(crime_router)

# 루트 경로
@app.get("/")
async def root():
    return {
        "service": "Crime Analysis Service",
        "version": "0.1.0"
    }

# 서버 실행
if __name__ == "__main__":
    port = int(os.getenv("PORT", 9002))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)