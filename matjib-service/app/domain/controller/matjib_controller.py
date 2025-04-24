from app.domain.service.matjib_serevice import MatjibService


class MatjibController:
    def __init__(self):
        self.service = MatjibService()

    def preprocess(self):
        print("🍽️ MatjibController: 전처리 호출")
        dataset = self.service.preprocess('matjib.csv')
        return dataset.matjib
    
    def learning(self):
        pass

    def evaluation(this):
        pass

    def submit(self):
        pass
        