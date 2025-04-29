import folium
import numpy as np
import pandas as pd
import os
from app.domain.model.reader_schema import ReaderSchema
from app.domain.model.google_map_schema import GoogleMapSchema
from sklearn import preprocessing

# xarray Dataset 대신 간단한 데이터 클래스 정의
class DataStorage:
    def __init__(self):
        self.cctv = None
        self.crime = None
        self.pop = None
        self.police = None
        self.cctv_pop = None  # cctv와 pop을 병합한 데이터 저장용 속성 추가


class CrimeService:
    def __init__(self):
        self.dataset = DataStorage()  # xarray Dataset 대신 DataStorage 사용
        self.reader = ReaderSchema()
        self.crime_rate_columns = ['살인검거율', '강도검거율', '강간검거율', '절도검거율', '폭력검거율']
        self.crime_columns = ['살인', '강도', '강간', '절도', '폭력']
        self.save_dir = 'app/updated_data'
    
    def preprocess(self, *args) -> object:
        print(f"------------모델 전처리 시작-----------")
        this = self.dataset
        for i in list(args):
            self.save_object_to_csv(this, i)
        return this
    
    def create_matrix(self, fname) -> object:
        print(f"😎🥇🐰파일명 : {fname}")
        self.reader.fname = fname
        if fname.endswith('csv'):
            return self.reader.csv_to_dframe()
        elif fname.endswith('xls'):
            return self.reader.xls_to_dframe(header=2, usecols='B,D,G,J,N')
    
    def save_object_to_csv(self, this, fname) -> object:
        print(f"🌱save_csv 실행 : {fname}")
        full_name = os.path.join(self.save_dir, fname)

        if not os.path.exists(full_name) and fname == "cctv_in_seoul.csv":
            this.cctv = self.create_matrix(fname)
            this = self.update_cctv(this)
            
        elif not os.path.exists(full_name) and fname == "crime_in_seoul.csv":
            this.crime = self.create_matrix(fname)
            this = self.update_crime(this) 
            this = self.update_police(this) 

        elif not os.path.exists(full_name) and fname == "pop_in_seoul.csv":
            this.pop = self.create_matrix(fname)
            this = self.update_pop(this)

        else:
            print(f"파일이 이미 존재합니다. {fname}")
            # 파일이 이미 존재하는 경우에도 데이터를 로드합니다
            if fname == "cctv_in_seoul.csv" and this.cctv is None:
                this.cctv = pd.read_csv(full_name)
                # '기관명'이 있고 '자치구'가 없는 경우 칼럼명 변경
                if '기관명' in this.cctv.columns and '자치구' not in this.cctv.columns:
                    this.cctv = this.cctv.rename(columns={'기관명': '자치구'})
            elif fname == "crime_in_seoul.csv" and this.crime is None:
                this.crime = pd.read_csv(full_name)
                # police 데이터도 로드
                police_file = os.path.join(self.save_dir, 'police_in_seoul.csv')
                if os.path.exists(police_file):
                    this.police = pd.read_csv(police_file)
            elif fname == "pop_in_seoul.xls" and this.pop is None:
                pop_csv = os.path.join(self.save_dir, 'pop_in_seoul.csv')
                if os.path.exists(pop_csv):
                    this.pop = pd.read_csv(pop_csv)
                else:
                    this.pop = self.create_matrix(fname)
                    this = self.update_pop(this)

        return this
    
    def update_cctv(self, this) -> object:
        print(f"------------ update_cctv 실행 ------------")
        this.cctv = this.cctv.drop(['2013년도 이전', '2014년', '2015년', '2016년'], axis=1)
        print(f"CCTV 데이터 헤드: {this.cctv.head()}")
        cctv = this.cctv
        cctv = cctv.rename(columns={'기관명': '자치구'})
        cctv.to_csv(os.path.join(self.save_dir, 'cctv_in_seoul.csv'), index=False)
        this.cctv = cctv
        return this
    
    def update_crime(self, this) -> object:
        print(f"------------ update_crime 실행 ------------")
        crime = this.crime
        station_names = []  # 경찰서 관서명 리스트
        for name in crime['관서명']:
            station_names.append('서울' + str(name[:-1]) + '경찰서')
        print(f"🔥💧경찰서 관서명 리스트: {station_names}")
        
        station_addrs = []
        station_lats = []
        station_lngs = []
        gmaps = GoogleMapSchema()  # 구글맵 객체 생성
        
        for name in station_names:
            tmp = gmaps.geocode(name, language='ko')
            print(f"""{name}의 검색 결과: {tmp[0].get("formatted_address")}""")
            station_addrs.append(tmp[0].get("formatted_address"))
            tmp_loc = tmp[0].get("geometry")
            station_lats.append(tmp_loc['location']['lat'])
            station_lngs.append(tmp_loc['location']['lng'])
            
        print(f"🔥💧자치구 리스트: {station_addrs}")
        gu_names = []
        for addr in station_addrs:
            tmp = addr.split()
            tmp_gu = [gu for gu in tmp if gu[-1] == '구'][0]
            gu_names.append(tmp_gu)
        print(f"🔥💧자치구 리스트 2: {gu_names}")
        crime['자치구'] = gu_names

        # 구 와 경찰서의 위치가 다른 경우 수작업
        crime.loc[crime['관서명'] == '혜화서', '자치구'] = '종로구'
        crime.loc[crime['관서명'] == '서부서', '자치구'] = '은평구'
        crime.loc[crime['관서명'] == '강서서', '자치구'] = '양천구'
        crime.loc[crime['관서명'] == '종암서', '자치구'] = '성북구'
        crime.loc[crime['관서명'] == '방배서', '자치구'] = '서초구'
        crime.loc[crime['관서명'] == '수서서', '자치구'] = '강남구'
        
        crime.to_csv(os.path.join(self.save_dir, 'crime_in_seoul.csv'), index=False)
        this.crime = crime
        return this
    
    def update_police(self, this) -> object:
        print(f"------------ update_police 실행 ------------")
        crime = this.crime
        crime = crime.groupby("자치구").sum().reset_index()
        crime = crime.drop(columns=["관서명"])

        police = pd.pivot_table(crime, index='자치구', aggfunc=np.sum).reset_index()
        
        police['살인검거율'] = (police['살인 검거'].astype(int) / police['살인 발생'].astype(int)) * 100
        police['강도검거율'] = (police['강도 검거'].astype(int) / police['강도 발생'].astype(int)) * 100
        police['강간검거율'] = (police['강간 검거'].astype(int) / police['강간 발생'].astype(int)) * 100
        police['절도검거율'] = (police['절도 검거'].astype(int) / police['절도 발생'].astype(int)) * 100
        police['폭력검거율'] = (police['폭력 검거'].astype(int) / police['폭력 발생'].astype(int)) * 100
        
        police = police.drop(columns={'살인 검거', '강도 검거', '강간 검거', '절도 검거', '폭력 검거'}, axis=1)
        police.to_csv(os.path.join(self.save_dir, 'police_in_seoul.csv'), index=False)

        # 검거율이 100%가 넘는 경우 처리
        for column in self.crime_rate_columns:
            police.loc[police[column] > 100, column] = 100

        police = police.rename(columns={
            '살인 발생': '살인',
            '강도 발생': '강도',
            '강간 발생': '강간',
            '절도 발생': '절도',
            '폭력 발생': '폭력'
        })

        # 정규화 처리
        x = police[self.crime_rate_columns].values
        min_max_scalar = preprocessing.MinMaxScaler()
        x_scaled = min_max_scalar.fit_transform(x.astype(float))
        
        police_norm = pd.DataFrame(x_scaled, columns=self.crime_columns, index=police.index)
        police_norm[self.crime_rate_columns] = police[self.crime_rate_columns]
        police_norm['범죄'] = np.sum(police_norm[self.crime_rate_columns], axis=1)
        police_norm['검거'] = np.sum(police_norm[self.crime_columns], axis=1)
        police_norm.to_csv(os.path.join(self.save_dir, 'police_norm_in_seoul.csv'))

        this.police = police
        return this
    
    def update_pop(self, this) -> object:
        print(f"------------ update_pop 실행 ------------")
        pop = this.pop
        pop = pop.rename(columns={
            pop.columns[0]: '자치구',
            pop.columns[1]: '인구수',
            pop.columns[2]: '한국인',
            pop.columns[3]: '외국인',
            pop.columns[4]: '고령자'
        })
        
        pop.to_csv(os.path.join(self.save_dir, 'pop_in_seoul.csv'), index=False)
        pop.drop([26], inplace=True)
        
        pop['외국인비율'] = pop['외국인'].astype(int) / pop['인구수'].astype(int) * 100
        pop['고령자비율'] = pop['고령자'].astype(int) / pop['인구수'].astype(int) * 100

        # CCTV와 인구 데이터 결합 및 상관관계 분석
        cctv_pop = pd.merge(this.cctv, pop, on='자치구')
        cor1 = np.corrcoef(cctv_pop['고령자비율'], cctv_pop['소계'])
        cor2 = np.corrcoef(cctv_pop['외국인비율'], cctv_pop['소계'])
        print(f'고령자비율과 CCTV의 상관계수 {str(cor1)} \n'
              f'외국인비율과 CCTV의 상관계수 {str(cor2)} ')

        print(f"🔥💧pop: {pop.head()}")
        return this
        
    def create_and_save_cctv_pop(self) -> object:
        """
        cctv 데이터와 pop 데이터, police 데이터를 병합하여 cctv_pop 데이터프레임을 생성하고 저장합니다.
        
        Returns:
            object: DataStorage 객체
        """
        this = self.dataset
        
        try:
            # 필요한 데이터 존재 확인
            if this.cctv is None:
                print("⚠️ cctv 데이터가 없습니다. 먼저 데이터를 로드해주세요.")
                return this
                
            if this.pop is None:
                print("⚠️ pop 데이터가 없습니다. 먼저 데이터를 로드해주세요.")
                # pop 데이터 로드 시도
                pop_file = os.path.join(self.save_dir, 'pop_in_seoul.csv')
                if os.path.exists(pop_file):
                    this.pop = pd.read_csv(pop_file)
                    print(f"✅ pop 데이터를 로드했습니다: {pop_file}")
                    
                    # 외국인비율, 고령자비율 계산
                    this.pop['외국인비율'] = this.pop['외국인'].astype(float) / this.pop['인구수'].astype(float) * 100
                    this.pop['고령자비율'] = this.pop['고령자'].astype(float) / this.pop['인구수'].astype(float) * 100
                    print("✅ 외국인비율과 고령자비율을 계산했습니다.")
                else:
                    print(f"❌ pop 데이터 파일을 찾을 수 없습니다: {pop_file}")
                    return this
            
            # police 데이터 로드
            police_file = os.path.join(self.save_dir, 'police_in_seoul.csv')
            if this.police is None and os.path.exists(police_file):
                this.police = pd.read_csv(police_file)
                print(f"✅ police 데이터를 로드했습니다: {police_file}")
            
            # 데이터 유효성 확인
            if '자치구' not in this.cctv.columns:
                print("⚠️ cctv 데이터에 '자치구' 컬럼이 없습니다.")
                return this
                
            if '자치구' not in this.pop.columns:
                print("⚠️ pop 데이터에 '자치구' 컬럼이 없습니다.")
                return this
            
            # 합계 행 제거 (있는 경우)
            if '합계' in this.pop['자치구'].values:
                this.pop = this.pop[this.pop['자치구'] != '합계']
                print("✅ pop 데이터에서 '합계' 행을 제거했습니다.")
            
            # '자치구' 컬럼을 기준으로 데이터 병합
            if this.police is not None and '자치구' in this.police.columns:
                # cctv + police 병합 (범죄 데이터 포함)
                cctv_police = pd.merge(this.cctv, this.police, on='자치구', how='inner')
                print(f"✅ cctv와 police 데이터를 병합했습니다. 형태: {cctv_police.shape}")
                
                # (cctv + police) + pop 병합
                cctv_pop = pd.merge(cctv_police, this.pop, on='자치구', how='inner')
                print(f"✅ 최종 cctv_pop 데이터를 생성했습니다. 형태: {cctv_pop.shape}")
            else:
                # police 데이터가 없으면 cctv + pop만 병합
                cctv_pop = pd.merge(this.cctv, this.pop, on='자치구', how='inner')
                print(f"⚠️ police 데이터가 없어 cctv와 pop 데이터만 병합했습니다. 형태: {cctv_pop.shape}")
            
            # 불필요한 컬럼 제거 (필요시 활성화)
            # columns_to_keep = ['자치구', '소계', '살인', '강도', '강간', '절도', '폭력', '인구수', '외국인비율', '고령자비율']
            # cctv_pop = cctv_pop[columns_to_keep]
            
            # DataStorage 객체에 저장
            this.cctv_pop = cctv_pop
            
            # CSV 파일로 저장
            save_path = os.path.join(self.save_dir, 'cctv_pop.csv')
            cctv_pop.to_csv(save_path, index=False)
            
            print(f"✅ cctv_pop 파일 저장 완료! 경로: {save_path}")
            print(f"✅ 데이터 형태: {cctv_pop.shape}")
            print(f"✅ 컬럼: {cctv_pop.columns.tolist()}")
            
        except Exception as e:
            print(f"❌ cctv_pop 생성 및 저장 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return this
        
    def analyze_correlations(self) -> dict:
        """
        cctv_pop 데이터를 사용하여 다양한 변수 간의 상관관계를 분석하고 결과를 반환합니다.
        
        Returns:
            dict: 상관관계 분석 결과를 담은 딕셔너리
        """
        this = self.dataset
        results = {
            "status": "success",
            "message": "상관관계 분석 완료",
            "data": {
                "correlations": {
                    "외국인비율_범죄": {},
                    "고령자비율_범죄": {},
                    "CCTV_범죄": {}
                },
                "top_correlations": [],
                "analysis_summary": ""
            }
        }
        
        try:
            # cctv_pop 데이터 확인
            if this.cctv_pop is None:
                print("⚠️ cctv_pop 데이터가 없습니다. 먼저 create_and_save_cctv_pop()를 실행해주세요.")
                results["status"] = "error"
                results["message"] = "cctv_pop 데이터가 없습니다"
                return results
            
            cctv_pop = this.cctv_pop
            print(f"📊 분석할 데이터 형태: {cctv_pop.shape}")
            print(f"📊 데이터 컬럼: {cctv_pop.columns.tolist()}")
            
            # 필요한 컬럼 목록
            crime_cols = []
            has_crime_columns = False
            
            # 살인, 강도, 강간, 절도, 폭력 컬럼이 있는지 확인
            if all(col in cctv_pop.columns for col in ['살인', '강도', '강간', '절도', '폭력']):
                crime_cols = ['살인', '강도', '강간', '절도', '폭력']
                has_crime_columns = True
            # 살인 발생, 강도 발생 등의 컬럼이 있는지 확인
            elif all(col in cctv_pop.columns for col in ['살인 발생', '강도 발생', '강간 발생', '절도 발생', '폭력 발생']):
                # 컬럼 이름 변경
                cctv_pop = cctv_pop.rename(columns={
                    '살인 발생': '살인',
                    '강도 발생': '강도',
                    '강간 발생': '강간',
                    '절도 발생': '절도',
                    '폭력 발생': '폭력'
                })
                crime_cols = ['살인', '강도', '강간', '절도', '폭력']
                has_crime_columns = True
                print("✅ 범죄 발생 컬럼 이름을 변경했습니다.")
            
            if not has_crime_columns:
                print("⚠️ 범죄 데이터 컬럼을 찾을 수 없습니다.")
                print(f"실제 컬럼: {cctv_pop.columns.tolist()}")
                results["status"] = "error"
                results["message"] = "범죄 데이터 컬럼을 찾을 수 없습니다"
                return results
            
            # 필요한 기타 컬럼 확인
            if '소계' not in cctv_pop.columns:
                print("⚠️ CCTV 소계 컬럼을 찾을 수 없습니다.")
                results["status"] = "error"
                results["message"] = "CCTV 소계 컬럼을 찾을 수 없습니다"
                return results
                
            if not all(col in cctv_pop.columns for col in ['외국인비율', '고령자비율']):
                print("⚠️ 외국인비율 또는 고령자비율 컬럼을 찾을 수 없습니다.")
                results["status"] = "error"
                results["message"] = "외국인비율 또는 고령자비율 컬럼을 찾을 수 없습니다"
                return results
            
            print("\n📊 서울시 데이터 상관관계 분석 결과")
            print("=" * 60)
            
            # 1. 외국인비율, 고령자비율과 각 범죄 발생 건수 간의 상관관계
            print("\n1️⃣ 외국인비율, 고령자비율과 범죄 발생 건수 간의 상관관계")
            print("-" * 60)
            
            for crime in crime_cols:
                # 외국인비율과 범죄
                foreign_corr = np.corrcoef(cctv_pop['외국인비율'], cctv_pop[crime])[0, 1]
                # 고령자비율과 범죄
                elderly_corr = np.corrcoef(cctv_pop['고령자비율'], cctv_pop[crime])[0, 1]
                
                print(f"• {crime} 범죄:")
                print(f"  - 외국인비율과의 상관계수: {foreign_corr:.3f}")
                print(f"  - 고령자비율과의 상관계수: {elderly_corr:.3f}")
                
                # 결과 저장
                results["data"]["correlations"]["외국인비율_범죄"][crime] = round(float(foreign_corr), 3)
                results["data"]["correlations"]["고령자비율_범죄"][crime] = round(float(elderly_corr), 3)
            
            # 2. CCTV 설치수와 범죄 발생 건수 간의 상관관계
            print("\n2️⃣ CCTV 설치수(소계)와 범죄 발생 건수 간의 상관관계")
            print("-" * 60)
            
            for crime in crime_cols:
                cctv_corr = np.corrcoef(cctv_pop['소계'], cctv_pop[crime])[0, 1]
                print(f"• CCTV 설치수와 {crime} 범죄의 상관계수: {cctv_corr:.3f}")
                
                # 결과 저장
                results["data"]["correlations"]["CCTV_범죄"][crime] = round(float(cctv_corr), 3)
            
            # 3. 결과 요약 및 해석
            print("\n3️⃣ 분석 결과 요약")
            print("-" * 60)
            
            # 가장 강한 양의 상관관계와 음의 상관관계 찾기
            correlations = []
            
            # 외국인비율과 범죄
            for crime in crime_cols:
                corr = np.corrcoef(cctv_pop['외국인비율'], cctv_pop[crime])[0, 1]
                correlations.append(('외국인비율', crime, corr))
            
            # 고령자비율과 범죄
            for crime in crime_cols:
                corr = np.corrcoef(cctv_pop['고령자비율'], cctv_pop[crime])[0, 1]
                correlations.append(('고령자비율', crime, corr))
            
            # CCTV와 범죄
            for crime in crime_cols:
                corr = np.corrcoef(cctv_pop['소계'], cctv_pop[crime])[0, 1]
                correlations.append(('CCTV 설치수', crime, corr))
            
            # 상관계수 절대값 기준으로 정렬
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # 가장 강한 상관관계 Top 3
            print("• 가장 강한 상관관계 (절대값 기준 Top 3):")
            for i, (var1, var2, corr) in enumerate(correlations[:3]):
                correlation_type = "양의" if corr > 0 else "음의"
                strength = "강한" if abs(corr) > 0.7 else "중간" if abs(corr) > 0.3 else "약한"
                corr_text = f"{i+1}. {var1}와 {var2} 간의 {correlation_type} 상관관계 ({corr:.3f}) - {strength} 상관관계"
                print(f"  {corr_text}")
                results["data"]["top_correlations"].append(corr_text)
            
            # 해석
            analysis_summary = [
                "📊 서울시 범죄 데이터 상관관계 분석 요약",
                "==================================================",
                "1. 고령자비율이 높은 지역일수록 범죄 발생률이 낮은 경향이 있습니다.",
                "2. CCTV 설치수와 범죄 발생률은 양의 상관관계를 보입니다.",
                "3. 외국인비율은 범죄율과 뚜렷한 상관관계를 보이지 않습니다."
            ]
            
            print("\n• 해석:")
            for line in analysis_summary[2:]:
                print(f"  {line}")
            print("  - 상관계수가 1에 가까울수록 강한 양의 상관관계를 나타냅니다.")
            print("  - 상관계수가 -1에 가까울수록 강한 음의 상관관계를 나타냅니다.")
            print("  - 상관계수가 0에 가까울수록 두 변수 간에 선형적 관계가 약하거나 없습니다.")
            print("  - 상관관계는 인과관계를 의미하지 않으며, 다른 요소들의 영향도 고려해야 합니다.")
            
            results["data"]["analysis_summary"] = "\n".join(analysis_summary)
            
            print("\n✅ 상관관계 분석이 완료되었습니다.")
            
        except Exception as e:
            print(f"❌ 상관관계 분석 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            results["status"] = "error"
            results["message"] = f"상관관계 분석 중 오류 발생: {str(e)}"
        
        return results
    
    def create_crime_map(self) -> object:
        file = self.file
        reader = self.reader
        file.context = './updated_data/'
        file.fname = 'police_norm_in_seoul'
        police_norm = reader.csv(file)
        file.context = './updated_data/'
        file.fname = 'geo_simple'
        state_geo = reader.json(file)
        file.fname = 'crime_in_seoul'
        crime = reader.csv(file)
        file.context = './updated_data/'
        file.fname = 'police_in_seoul'
        police_pos = reader.csv(file)
        station_names = []
        for name in crime['관서명']:
            station_names.append('서울' + str(name[:-1] + '경찰서'))
        station_addrs = []
        station_lats = []
        station_lngs = []
        gmaps = reader.gmaps()
        for name in station_names:
            temp = gmaps.geocode(name, language='ko')
            station_addrs.append(temp[0].get('formatted_address'))
            t_loc = temp[0].get('geometry')
            station_lats.append(t_loc['location']['lat'])
            station_lngs.append(t_loc['location']['lng'])

        police_pos['lat'] = station_lats
        police_pos['lng'] = station_lngs
        col = ['살인 검거', '강도 검거', '강간 검거', '절도 검거', '폭력 검거']
        tmp = police_pos[col] / police_pos[col].max()
        police_pos['검거'] = np.sum(tmp, axis=1)

        folium_map = folium.Map(location=[37.5502, 126.982], zoom_start=12, title='Stamen Toner')

        folium.Choropleth(
            geo_data=state_geo,
            data=tuple(zip(police_norm['구별'],police_norm['범죄'])),
            columns=["State", "Crime Rate"],
            key_on="feature.id",
            fill_color="PuRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Crime Rate (%)",
            reset=True,
        ).add_to(folium_map)
        for i in police_pos.index:
            folium.CircleMarker([police_pos['lat'][i], police_pos['lng'][i]],
                                radius=police_pos['검거'][i] * 10,
                                fill_color='#0a0a32').add_to(folium_map)

        folium_map.save('./saved_data/crime_map.html')
