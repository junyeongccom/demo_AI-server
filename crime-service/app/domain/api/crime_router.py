from fastapi import APIRouter, BackgroundTasks
from typing import Dict, Any
from app.domain.controller.crime_controller import CrimeController
from app.domain.model.crime_schema import AnalysisResponse

# 라우터 생성
router = APIRouter(
    prefix="/crime",
    tags=["crime"],
    responses={404: {"description": "Not found"}}
)

# 컨트롤러 인스턴스 생성
controller = CrimeController()

# 루트 경로
@router.get("/")
async def root():
    return {
        "message": "Crime Analysis API", 
        "version": "0.1.0",
        "endpoints": [
            "/analyze - 전체 데이터 분석 실행",
            "/correlations - 데이터 분석 결과 조회"
        ]
    }

# 데이터 분석 실행 엔드포인트 (백그라운드로 실행)
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(background_tasks: BackgroundTasks):
    # 백그라운드에서 전처리 및 분석 실행
    def run_analysis():
        controller.preprocess()
        print("✅ 데이터 분석 완료")
    
    background_tasks.add_task(run_analysis)
    
    return {
        "status": "started", 
        "message": "데이터 분석이 시작되었습니다. 결과는 /correlations 엔드포인트에서 확인할 수 있습니다.",
        "data": None
    }

# 분석 결과 조회 엔드포인트
@router.get("/correlations", response_model=AnalysisResponse)
async def get_correlations():
    # 컨트롤러에서 분석 실행 및 결과 반환
    results = controller.preprocess()
    return results
