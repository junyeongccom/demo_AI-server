import pandas as pd
import os
from pathlib import Path
from app.domain.model.crime_schema import CrimeSchema
from app.domain.model.google_schema import ApiKeyManager
from app.domain.model.reader_schema import ReaderSchema

UPDATED_DIR = './app/updated_data'

class CrimeService:
    def __init__(self, data_dir: str = './app/store/'):
        self.dataset = CrimeSchema(cctv=None, crime=None, pop=None)
        self.datareader = ReaderSchema()
        self.data_dir = data_dir

    def load_data(self, fname: str) -> pd.DataFrame:
        # 기본 파일 경로 (csv 먼저 시도)
        path_csv = Path(self.data_dir) / fname
        path_xls = path_csv.with_suffix('.xls')  # 같은 이름의 .xls 경로

        if path_csv.exists():
            print(f"📂 CSV 파일 로드: {path_csv.name}")
            return pd.read_csv(path_csv)

        elif path_xls.exists():
            print(f"📂 Excel 파일 로드: {path_xls.name}")
            return pd.read_excel(path_xls, header=2, usecols='B,D,G,J,N')

        else:
            raise FileNotFoundError(f"❌ '{fname}' 또는 '{path_xls.name}' 파일이 존재하지 않습니다.")

    def preprocess(self, cctv_file: str, crime_file: str, pop_file: str) -> CrimeSchema:
        print("------------ 모델 전처리 시작 -----------")

        self.dataset.cctv = self.load_data(cctv_file)
        self.dataset.crime = self.load_data(crime_file)
        self.dataset.pop = self.load_data(pop_file)

        self.dataset = self.update_cctv(self.dataset)
        self.dataset = self.update_crime(self.dataset)
        self.dataset = self.update_pop(self.dataset)

        return self.dataset

    def update_cctv(self, dataset: CrimeSchema) -> CrimeSchema:
        cctv = dataset.cctv.drop(columns=['2013년도 이전', '2014년', '2015년', '2016년'])
        cctv = cctv.rename(columns={'기관명': '자치구'})
        self.save_to_updated_data(cctv, "cctv_seoul.csv")
        print(f"😄 CCTV 데이터 확인:\n{cctv.head()}")
        dataset.cctv = cctv
        return dataset

    def update_crime(self, dataset: CrimeSchema) -> CrimeSchema:
        crime = dataset.crime
        station_names = ['서울' + str(name[:-1]) + '경찰서' for name in crime['관서명']]
        print(f"😄 관서명 리스트: {station_names}")

        gmaps = ApiKeyManager()
        station_addrs, station_lats, station_lngs = [], [], []

        for name in station_names:
            tmp = gmaps.geocode(name, language='ko')
            station_addrs.append(tmp[0].get("formatted_address"))
            print(f"{name}의 주소: {tmp[0].get('formatted_address')}")
            tmp_loc = tmp[0].get("geometry")
            station_lats.append(tmp_loc['location']['lat'])
            station_lngs.append(tmp_loc['location']['lng'])

        gu_names = [addr.split()[[gu[-1] == '구' for gu in addr.split()].index(True)] for addr in station_addrs]
        crime['자치구'] = gu_names
        crime = self.crime_modify(crime)

        self.save_to_updated_data(crime, 'crime_seoul.csv')
        dataset.crime = crime
        return dataset

    def crime_modify(self, crime_df: pd.DataFrame) -> pd.DataFrame:
        crime_df["발생 합계"] = crime_df.filter(like="발생").sum(axis=1)
        crime_df["검거 합계"] = crime_df.filter(like="검거").sum(axis=1)
        crime_df = crime_df[["자치구", "발생 합계", "검거 합계"]]
        print(f"😄 범죄 데이터 수정 확인:\n{crime_df.head()}")
        return crime_df

    def update_pop(self, dataset: CrimeSchema) -> CrimeSchema:
        pop = dataset.pop
        pop = pop.rename(columns={
            pop.columns[1]: '인구수',
            pop.columns[2]: '한국인',
            pop.columns[3]: '외국인',
            pop.columns[4]: '고령자'
        })
        self.save_to_updated_data(pop, 'pop_seoul.csv')
        print(f"😄 인구 데이터 확인:\n{pop.head()}")
        dataset.pop = pop
        return dataset

    def save_to_updated_data(self, df: pd.DataFrame, filename: str):
        os.makedirs(UPDATED_DIR, exist_ok=True)
        full_path = os.path.join(UPDATED_DIR, filename)
        df.to_csv(full_path, index=False)
        print(f"💾 저장 완료: {full_path}")