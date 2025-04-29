from app.domain.service.crime_service import CrimeService
from typing import Dict, Any


class CrimeController:
    def __init__(self):
        self.service = CrimeService()

    def preprocess(self) -> Dict[str, Any]:
        """
        데이터 전처리 및 분석을 수행하고 결과를 반환합니다.
        
        Returns:
            Dict[str, Any]: 분석 결과를 담은 딕셔너리
        """
        try:
            # 1. 기본 데이터 전처리
            self.service.preprocess('cctv_in_seoul.csv', 'crime_in_seoul.csv', 'pop_in_seoul.csv')
            
            # 2. CCTV와 인구 데이터 병합
            self.service.create_and_save_cctv_pop()
            
            # 3. 상관관계 분석 수행 및 결과 반환
            analysis_results = self.service.analyze_correlations()
            
            return analysis_results
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"분석 중 오류 발생: {str(e)}",
                "data": {}
            }
    
    def learning(self):
        pass

    def evaluation(this):
        pass

    def submit(self):
        pass