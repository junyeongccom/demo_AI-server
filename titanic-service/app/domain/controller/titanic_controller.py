from app.domain.service.titanic_service import TitanicService
from app.domain.model.titanic_schema import TitanicSchema

'''
print(f'결정트리 활용한 검증 정확도 {None}')
print(f'랜덤포레스트 활용한 검증 정확도 {None}')
print(f'나이브베이즈 활용한 검증 정확도 {None}')
print(f'KNN 활용한 검증 정확도 {None}')
print(f'SVM 활용한 검증 정확도 {None}')
'''
class TitanicController:
    def __init__(self):
        self.titanic_service = TitanicService()

    def preprocess(self) -> TitanicSchema:
        print("📦 전처리 시작 (Service 내부에서 전체 수행)")
        # 경로, 파일명은 내부에서 설정하거나 인자로 전달
        ds = self.titanic_service.preprocess('train.csv', 'test.csv')
        print("✅ 전처리 완료")
        return ds
    
    def learning(self):
        pass

    def evaluation(this):
        pass

    def submit(self):
        pass

    def find_best_model(self):
        return self.titanic_service.find_best_model()

