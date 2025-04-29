import json
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging

from app.domain.controller.titanic_controller import TitanicController

# 로거 설정
logger = logging.getLogger("titanic-service")

# 라우터 생성
titanic_router = APIRouter(tags=["titanic"])

# 컨트롤러 인스턴스 생성
titanic_controller = TitanicController()

# 기본 경로
@titanic_router.get("/")
async def root():
    return {"message": "Titanic Service API", "version": "0.1.0"}

# 상태 확인
@titanic_router.get("/health")
async def health_check():
    return {"status": "ok", "service": "titanic"}

# 모델 정보 조회
@titanic_router.get("/model-info")
async def get_model_info():
    try:
        # 전처리 데이터셋 준비
        dataset = titanic_controller.preprocess()
        # 학습 및 평가
        sorted_results, best_model, best_accuracy = titanic_controller.evaluation()
        
        return {
            "model": best_model,
            "accuracy": best_accuracy,
            "features": dataset.train.columns.tolist()
        }
    except Exception as e:
        logger.error(f"모델 정보 조회 중 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"모델 정보 조회 중 오류 발생: {str(e)}"}
        )

# 예측 실행
@titanic_router.post("/predict")
async def predict(request: Request):
    try:
        # 요청 바디 읽기
        body = await request.json()
        
        # 전처리 데이터셋 준비 (실제 애플리케이션에서는 이 부분이 서버 시작 시 한 번만 실행되어야 함)
        dataset = titanic_controller.preprocess()
        
        # 여기서 실제 예측 코드 구현 필요
        # 지금은 더미 데이터 반환
        prediction_result = {
            "request": body,
            "prediction": {
                "survived": 1,
                "probability": 0.85
            },
            "model": "RandomForest"
        }
        
        return prediction_result
    except Exception as e:
        logger.error(f"예측 중 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"예측 중 오류 발생: {str(e)}"}
        )

# 데이터셋 생성 및 저장
@titanic_router.post("/generate-submission")
async def generate_submission():
    try:
        # 전처리 데이터셋 준비
        dataset = titanic_controller.preprocess()
        # 예측 결과 생성 및 파일 저장
        submission = titanic_controller.submit()
        
        return {
            "status": "success",
            "message": "예측 결과 파일이 생성되었습니다."
        }
    except Exception as e:
        logger.error(f"제출 파일 생성 중 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"제출 파일 생성 중 오류 발생: {str(e)}"}
        )
