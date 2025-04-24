from app.domain.service.titanic_service import TitanicService


class TitanicController:
    def __init__(self):
        self.service = TitanicService()

    def preprocess(self):
        print("📦 전처리 시작 (Service 내부에서 전체 수행)")
        # 경로, 파일명은 내부에서 설정하거나 인자로 전달
        dataset = self.service.preprocess('train.csv', 'test.csv')
        print("✅ 전처리 완료")
        return dataset
    
    def learning(self):
        pass

    def evaluation(this):
        pass

    def submit(self):
        pass

