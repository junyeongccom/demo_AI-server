import pandas as pd
from pathlib import Path
from app.domain.model.matjib_schema import MatjibSchema

class MatjibService:
    def __init__(self, data_dir: str = './app/store'):
        self.dataset = MatjibSchema(
            context=data_dir,
            fname='',
            matjib=None,
            id='',
            name=''
        )

    def load_data(self, fname: str) -> pd.DataFrame:
        path = Path(self.dataset.context) / fname
        self.dataset.fname = fname
        df = pd.read_csv(path)
        self.dataset.matjib = df
        return df

    def preprocess(self, fname: str = 'matjib.csv') -> MatjibSchema:
        print("🍽️ 맛집 데이터 전처리 시작")
        self.load_data(fname)
        df = self.dataset.matjib

        # 예시 전처리
        if 'unnecessary_col1' in df.columns:
            df.drop(columns='unnecessary_col1', inplace=True)

        df.fillna(0, inplace=True)

        print("✅ 전처리 완료")
        return self.dataset
