from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import sys
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# 타이타닉 라우터 임포트
from app.domain.api.titanic_router import titanic_router

# ✅ 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("titanic-service")

# ✅ .env 파일 로드
load_dotenv()

# ✅ 애플리케이션 시작 시 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 타이타닉 서비스 시작")
    yield
    logger.info("🛑 타이타닉 서비스 종료")

# ✅ FastAPI 앱 생성
app = FastAPI(
    title="Titanic API",
    description="Titanic prediction service for conan.ai.kr",
    version="0.1.0",
    lifespan=lifespan
)

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 환경에서는 구체적인 도메인 지정 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 메인 라우터 생성 및 등록
app.include_router(titanic_router, prefix="/titanic")

# 기본 루트 경로
@app.get("/")
async def root():
    return {"message": "Titanic Service API", "version": "0.1.0"}

# ✅ 서버 실행
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    logger.info(f"타이타닉 서비스 시작: 포트 {port}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
