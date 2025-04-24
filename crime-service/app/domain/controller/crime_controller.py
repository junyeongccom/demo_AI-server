from app.domain.service.crime_service import CrimeService


class CrimeController:
    def __init__(self):
        self.service = CrimeService()

    def preprocess(self):
        dataset = self.service.preprocess('cctv_in_seoul.csv', 'crime_in_seoul.csv', 'pop_in_seoul.csv')
        return dataset
    
    def learning(self):
        pass

    def evaluation(this):
        pass

    def submit(self):
        pass