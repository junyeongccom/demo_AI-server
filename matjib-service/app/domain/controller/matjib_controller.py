from app.domain.service.matjib_serevice import MatjibService


class MatjibController:
    def __init__(self):
        self.service = MatjibService()

    def preprocess(self):
        print("🍽️ MatjibController: 전처리 호출")
        dataset = self.service.preprocess('matjib.csv')
        df = dataset.matjib  # 전처리된 DataFrame

        # 🔍 디버깅 로그 출력
        print('*' * 100)
        print(f'1. DataFrame 타입: {type(df)}')
        print(f'2. 컬럼 목록:\n{df.columns.tolist()}')
        print(f'3. 상위 1개 행:\n{df.head(1)}')
        print(f'4. 결측값 개수:\n{df.isnull().sum()}개')
        print('*' * 100)

        return df
    
    def learning(self):
        pass

    def evaluation(this):
        pass

    def submit(self):
        pass
        