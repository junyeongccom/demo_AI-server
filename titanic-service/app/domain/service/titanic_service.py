import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
# XGBoost, LightGBM 라이브러리 추가
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    print("⚠️ XGBoost 라이브러리가 설치되어 있지 않습니다. 'pip install xgboost' 명령으로 설치해주세요.")
    XGBClassifier = None

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    print("⚠️ LightGBM 라이브러리가 설치되어 있지 않습니다. 'pip install lightgbm' 명령으로 설치해주세요.")
    LGBMClassifier = None

from app.domain.model.titanic_schema import TitanicSchema

"""
PassengerId  고객ID,
Survived 생존여부,
Pclass 승선권 1 = 1등석, 2 = 2등석, 3 = 3등석,
Name,
Sex,
Age,
SibSp 동반한 형제, 자매, 배우자,
Parch 동반한 부모, 자식,
Ticket 티켓번호,
Fare 요금,
Cabin 객실번호,
Embarked 승선한 항구명 C = 쉐브루, Q = 퀸즈타운, S = 사우스햄튼
"""

class TitanicService:
    def __init__(self, data_dir: str = './app/store/'):
        self.dataset = TitanicSchema(
            context=data_dir,
            fname='',
            train=None,
            test=None,
            id='',
            label=''
        )
        self.test_ids = None

    def new_model(self, fname: str) -> pd.DataFrame:
        path = Path(self.dataset.context) / fname
        return pd.read_csv(path)

    def preprocess(self, train_fname: str, test_fname: str, scaler_type: str = 'standard') -> TitanicSchema:
        """
        타이타닉 데이터 전처리를 수행합니다.
        
        Args:
            train_fname: 학습 데이터 파일명
            test_fname: 테스트 데이터 파일명
            scaler_type: 정규화 방식 ('standard' 또는 'minmax')
            
        Returns:
            전처리된 TitanicSchema 객체
        """
        print("-------- 모델 전처리 시작 --------")
        ds = self.dataset
        ds.train = self.new_model(train_fname)
        ds.test = self.new_model(test_fname)
        ds.id = ds.test['PassengerId']  # 예측 결과 제출용으로 ID 저장
        self.test_ids = ds.test['PassengerId']  # test_ids 속성 설정
        ds.label = ds.train['Survived']
        ds.train = ds.train.drop('Survived', axis=1)
        
        # PassengerId 제거 (학습에 불필요)
        ds = self.drop_feature(ds, 'PassengerId')
        
        # FamilySize 및 IsAlone 특성 추가
        ds = self.add_family_features(ds)
        
        # Name에서 Title 추출
        ds = self.extract_title_from_name(ds)
        title_mapping = self.remove_duplicate_title(ds)
        ds = self.title_nominal(ds, title_mapping)
        ds = self.drop_feature(ds, 'Name')
        
        # 성별(Sex) 처리
        ds = self.gender_nominal(ds)
        ds = self.drop_feature(ds, 'Sex')
        
        # 탑승항(Embarked) 처리
        ds = self.embarked_nominal(ds)
        
        # Age 결측치 처리 (Pclass와 Sex 기반)
        ds = self.fill_age_by_group(ds)
        
        # Fare 결측치 처리 (Pclass 기반)
        ds = self.fill_fare_by_pclass(ds)
        
        # 추가 파생변수 생성 (Age, Fare 원본 값이 필요하므로 drop 전에 추가)
        ds = self.add_advanced_features(ds)
        
        # Age, Fare 구간화
        ds = self.age_ratio_enhanced(ds)
        ds = self.fare_ratio_enhanced(ds)
        
        # 원본 Age, Fare 컬럼 제거
        ds = self.drop_feature(ds, 'Age', 'Fare')
        
        # Pclass 처리
        ds = self.pclass_ordinal(ds)
        
        # 불필요한 원본 컬럼 제거
        ds = self.drop_feature(ds, 'SibSp', 'Parch', 'Cabin', 'Ticket')
        
        # 특성 조합(Feature Interaction) 추가
        ds = self.add_feature_interactions(ds)
        
        # 범주형 변수 원-핫 인코딩
        ds = self.apply_one_hot_encoding(ds)
        
        # 정규화
        ds = self.normalize_features(ds, scaler_type=scaler_type)
        
        # 최종 데이터셋 컬럼 확인
        print("\n🔍 최종 전처리 데이터셋 컬럼 목록:")
        print(ds.train.columns.tolist())

        return ds

    def drop_feature(self, dataset: TitanicSchema, *features: str) -> TitanicSchema:
        """특정 컬럼을 train/test 데이터셋에서 제거합니다."""
        for col in features:
            if col in dataset.train.columns:
                dataset.train.drop(columns=col, inplace=True)
            if col in dataset.test.columns:
                dataset.test.drop(columns=col, inplace=True)
        return dataset
    
    def add_family_features(self, dataset: TitanicSchema) -> TitanicSchema:
        """SibSp와 Parch를 이용해 FamilySize와 IsAlone 특성을 추가합니다."""
        for df in [dataset.train, dataset.test]:
            # 가족 크기 = 본인 + 형제/배우자 + 부모/자녀
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            # 혼자 여행하는지 여부
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        print("👪 가족 관련 특성 추가 완료")
        return dataset

    def extract_title_from_name(self, dataset: TitanicSchema) -> TitanicSchema:
        """이름에서 타이틀(직함) 추출"""
        for df in [dataset.train, dataset.test]:
            df['Title'] = df['Name'].str.extract('([A-Za-z]+)\\.', expand=False)
        return dataset

    def remove_duplicate_title(self, dataset: TitanicSchema) -> dict:
        """타이틀 중복 제거하고 매핑 생성"""
        all_titles = set(dataset.train['Title'].unique()) | set(dataset.test['Title'].unique())
        print(f"📌 전체 직함 목록: {sorted(all_titles)}")
        return {
            'Mr': 1, 'Ms': 2, 'Mrs': 3, 'Master': 4,
            'Royal': 5, 'Rare': 6
        }

    def title_nominal(self, dataset: TitanicSchema, title_mapping: dict) -> TitanicSchema:
        """타이틀을 그룹화하고 숫자로 변환"""
        for df in [dataset.train, dataset.test]:
            df['Title'] = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            df['Title'] = df['Title'].replace(
                ['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Mme'], 'Rare')
            df['Title'] = df['Title'].replace({'Mlle': 'Mr', 'Miss': 'Ms'})
            df['Title'] = df['Title'].map(title_mapping)
        return dataset

    def gender_nominal(self, dataset: TitanicSchema) -> TitanicSchema:
        """성별을 숫자로 변환"""
        mapping = {'male': 1, 'female': 2}
        for df in [dataset.train, dataset.test]:
            df['Gender'] = df['Sex'].map(mapping)
        return dataset

    def embarked_nominal(self, dataset: TitanicSchema) -> TitanicSchema:
        """탑승항을 숫자로 변환하고 결측치는 최빈값으로 채움"""
        mapping = {'S': 1, 'C': 2, 'Q': 3}
        for df in [dataset.train, dataset.test]:
            # 결측치는 최빈값 'S'로 채움
            df['Embarked'] = df['Embarked'].fillna('S')
            df['Embarked'] = df['Embarked'].map(mapping)
        return dataset
    
    def fill_age_by_group(self, dataset: TitanicSchema) -> TitanicSchema:
        """Pclass와 Sex 조합 그룹별 중앙값으로 Age 결측치를 채웁니다."""
        # train과 test 데이터를 임시로 합쳐서 그룹별 통계값 계산
        combined_df = pd.concat([dataset.train, dataset.test])[['Age', 'Pclass', 'Gender']]
        
        # 그룹별 중앙값 계산
        age_medians = combined_df.groupby(['Pclass', 'Gender'])['Age'].median()
        print("📊 Pclass/Gender별 나이 중앙값:")
        print(age_medians)
        
        # 결측치 채우기
        for df in [dataset.train, dataset.test]:
            for (pclass, gender), age_median in age_medians.items():
                mask = (df['Age'].isna()) & (df['Pclass'] == pclass) & (df['Gender'] == gender)
                df.loc[mask, 'Age'] = age_median
                
            # 여전히 남아있는 결측치는 전체 중앙값으로 채움
            overall_median = combined_df['Age'].median()
            df['Age'] = df['Age'].fillna(overall_median)
        
        print("👴 나이 결측치 처리 완료")
        return dataset

    def age_ratio_enhanced(self, dataset: TitanicSchema) -> TitanicSchema:
        """나이를 더 세분화된 구간으로 나누어 매핑합니다 (총 10개 구간)."""
        mapping = {
            'Infant': 1,           # 0-2세
            'Toddler': 2,          # 3-5세
            'Child': 3,            # 6-12세
            'Teenager': 4,         # 13-18세
            'YoungAdult': 5,       # 19-25세
            'Adult': 6,            # 26-35세
            'MidAgeAdult': 7,      # 36-45세
            'MiddleAged': 8,       # 46-55세
            'Senior': 9,           # 56-65세
            'Elderly': 10          # 66세 이상
        }
        
        # 나이 구간 경계 정의
        bins = [0, 2, 5, 12, 18, 25, 35, 45, 55, 65, 100]
        labels = list(mapping.keys())

        for df in [dataset.train, dataset.test]:
            df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
            df['AgeGroup'] = df['AgeGroup'].map(mapping)
        
        print("👶 나이 그룹 세분화 완료 (10개 구간)")
        return dataset
    
    def fill_fare_by_pclass(self, dataset: TitanicSchema) -> TitanicSchema:
        """Pclass별 평균 요금으로 Fare 결측치를 채웁니다."""
        # train과 test 데이터를 임시로 합쳐서 그룹별 통계값 계산
        combined_df = pd.concat([dataset.train, dataset.test])[['Fare', 'Pclass']]
        
        # Pclass별 평균 요금 계산
        fare_means = combined_df.groupby('Pclass')['Fare'].mean()
        print("💲 객실등급별 평균 요금:")
        print(fare_means)
        
        # 결측치 채우기
        for df in [dataset.train, dataset.test]:
            for pclass, fare_mean in fare_means.items():
                mask = (df['Fare'].isna()) & (df['Pclass'] == pclass)
                df.loc[mask, 'Fare'] = fare_mean
                
            # 여전히 남아있는 결측치는 전체 평균으로 채움
            overall_mean = combined_df['Fare'].mean()
            df['Fare'] = df['Fare'].fillna(overall_mean)
        
        print("💰 요금 결측치 처리 완료")
        return dataset

    def fare_ratio_enhanced(self, dataset: TitanicSchema) -> TitanicSchema:
        """요금을 더 세분화된 구간으로 나누고, 상위 1% 럭셔리 구간을 분리합니다 (총 6개 구간)."""
        mapping = {
            'Free': 1,        # 무료
            'VeryLow': 2,     # 매우 저렴
            'Low': 3,         # 저렴
            'Mid': 4,         # 중간
            'High': 5,        # 고가
            'Luxury': 6       # 럭셔리 (상위 1%)
        }
        
        # 전체 데이터의 Fare 통계 계산
        combined_fare = pd.concat([dataset.train['Fare'], dataset.test['Fare']])
        
        # 상위 1% 경계값 계산
        luxury_threshold = combined_fare.quantile(0.99)
        
        # 6개 구간으로 나누기 위한 경계값 설정
        bins = [0, 0.01, 7.5, 15, 30, luxury_threshold, combined_fare.max()]
        labels = list(mapping.keys())

        for df in [dataset.train, dataset.test]:
            df['FareGroup'] = pd.cut(df['Fare'], bins=bins, labels=labels, include_lowest=True)
            df['FareGroup'] = df['FareGroup'].map(mapping)
        
        print(f"💳 요금 그룹 세분화 완료 (6개 구간, 럭셔리 기준: {luxury_threshold:.2f})")
        return dataset

    def pclass_ordinal(self, dataset: TitanicSchema) -> TitanicSchema:
        """객실등급 처리 (이미 숫자이므로 변경 없음)"""
        return dataset
    
    def add_feature_interactions(self, dataset: TitanicSchema) -> TitanicSchema:
        """특성 조합을 통해 새로운 특성을 생성합니다."""
        for df in [dataset.train, dataset.test]:
            # Sex와 Pclass 조합
            df['Gender_Pclass'] = df['Gender'].astype(str) + '_' + df['Pclass'].astype(str)
            gender_pclass_mapping = {val: idx for idx, val in enumerate(sorted(df['Gender_Pclass'].unique()), 1)}
            df['Gender_Pclass'] = df['Gender_Pclass'].map(gender_pclass_mapping)
            
            # AgeGroup과 Pclass 조합
            df['AgeGroup_Pclass'] = df['AgeGroup'].astype(str) + '_' + df['Pclass'].astype(str)
            age_pclass_mapping = {val: idx for idx, val in enumerate(sorted(df['AgeGroup_Pclass'].unique()), 1)}
            df['AgeGroup_Pclass'] = df['AgeGroup_Pclass'].map(age_pclass_mapping)
            
            # AgeGroup과 FareGroup 조합 (추가)
            df['AgeGroup_FareGroup'] = df['AgeGroup'].astype(str) + '_' + df['FareGroup'].astype(str)
            age_fare_mapping = {val: idx for idx, val in enumerate(sorted(df['AgeGroup_FareGroup'].unique()), 1)}
            df['AgeGroup_FareGroup'] = df['AgeGroup_FareGroup'].map(age_fare_mapping)
        
        print("🔄 특성 조합 추가 완료")
        return dataset
    
    def apply_one_hot_encoding(self, dataset: TitanicSchema) -> TitanicSchema:
        """범주형 변수에 원-핫 인코딩을 적용합니다."""
        # 원-핫 인코딩할 범주형 특성
        categorical_features = ['Pclass', 'Title', 'Embarked']
        
        # scikit-learn 버전 확인 및 OneHotEncoder 파라미터 설정
        try:
            current_version = version.parse(sklearn_version)
            version_boundary = version.parse("1.2")
            
            # 버전에 따라 적절한 인코더 선택
            if current_version >= version_boundary:
                # 1.2 이상 버전: sparse_output 사용
                print(f"🔍 scikit-learn 버전 {sklearn_version} 감지: sparse_output=False 사용")
                encoder = OneHotEncoder(sparse_output=False, drop='first')
            else:
                # 1.1 이하 버전: sparse 사용
                print(f"🔍 scikit-learn 버전 {sklearn_version} 감지: sparse=False 사용")
                encoder = OneHotEncoder(sparse=False, drop='first')
        except Exception as e:
            # 버전 비교 실패 시 안전하게 fallback
            print(f"⚠️ 버전 확인 중 오류 발생: {e}. sparse=False로 fallback")
            encoder = OneHotEncoder(sparse=False, drop='first')
        
        # train 데이터로 인코더 학습
        encoder.fit(dataset.train[categorical_features])
        
        # 변환 및 새 컬럼 추가
        for df_name, df in [('train', dataset.train), ('test', dataset.test)]:
            encoded_array = encoder.transform(df[categorical_features])
            
            # 생성된 원-핫 인코딩 컬럼에 이름 부여
            feature_names = []
            for i, feature in enumerate(categorical_features):
                categories = encoder.categories_[i][1:]  # 첫 번째 카테고리는 drop='first'로 제외됨
                feature_names.extend([f"{feature}_{category}" for category in categories])
            
            # 인코딩 결과를 데이터프레임으로 변환하여 원본에 추가
            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
            
            # 원본 데이터프레임에 병합
            result_df = pd.concat([df, encoded_df], axis=1)
            
            # 원래 범주형 특성은 제거
            result_df.drop(columns=categorical_features, inplace=True)
            
            # 결과 다시 할당
            if df_name == 'train':
                dataset.train = result_df
            else:
                dataset.test = result_df
        
        print("🔢 범주형 변수 원-핫 인코딩 완료")
        return dataset

    def normalize_features(self, dataset: TitanicSchema, scaler_type: str = 'standard') -> TitanicSchema:
        """
        수치형 특성을 정규화합니다.
        
        Args:
            dataset: 정규화할 데이터셋
            scaler_type: 'standard'(StandardScaler) 또는 'minmax'(MinMaxScaler)
        
        Returns:
            정규화된 데이터셋
        """
        # 스케일러 객체 선택
        if scaler_type.lower() == 'minmax':
            scaler = MinMaxScaler()
            print("🔄 MinMaxScaler로 데이터 정규화 진행")
        else:
            scaler = StandardScaler()
            print("🔄 StandardScaler로 데이터 정규화 진행")
        
        # 정규화할 수치형 특성 선택 (원-핫 인코딩된 변수는 제외)
        exclude_cols = ['Gender_Pclass', 'AgeGroup_Pclass', 'AgeGroup_FareGroup']  # 이미 인코딩된 상호작용 변수
        numeric_features = dataset.train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        
        if not numeric_features:
            print("⚠️ 정규화할 수치형 특성이 없습니다.")
            return dataset
        
        try:
            # 학습 데이터로 스케일러 학습 및 변환
            dataset.train[numeric_features] = scaler.fit_transform(dataset.train[numeric_features])
            
            # 테스트 데이터에 같은 스케일러 적용
            dataset.test[numeric_features] = scaler.transform(dataset.test[numeric_features])
            
            print(f"✅ 데이터 정규화 완료 (총 {len(numeric_features)}개 특성)")
        except Exception as e:
            print(f"⚠️ 데이터 정규화 중 오류 발생: {e}")
            print("정규화를 건너뛰고 원본 데이터를 사용합니다.")
        
        return dataset

    def add_advanced_features(self, dataset: TitanicSchema) -> TitanicSchema:
        """
        고급 파생변수를 추가합니다:
        1. Age × Pclass → Age_Pclass
        2. Fare ÷ FamilySize → Fare_Per_Person
        3. AgeGroup × FareGroup 조합 → AgeGroup_FareGroup (add_feature_interactions 메서드에서 추가)
        """
        for df in [dataset.train, dataset.test]:
            # Age × Pclass
            df['Age_Pclass'] = df['Age'] * df['Pclass']
            
            # Fare ÷ FamilySize (1인당 지불요금)
            # 0으로 나누는 것을 방지하기 위해 FamilySize가 0인 경우 1로 대체
            df['Fare_Per_Person'] = df['Fare'] / df['FamilySize'].replace(0, 1)
        
        print("🔍 고급 파생변수 추가 완료")
        return dataset
    
    def select_important_features(self, dataset: TitanicSchema) -> TitanicSchema:
        """
        RandomForest 기반으로 중요 변수만 선택합니다.
        중요도가 0.01 미만인 변수는 제거합니다.
        """
        # 학습 데이터에 적용
        if 'train' not in self.__dict__ or self.dataset.train is None or self.dataset.label is None:
            print("⚠️ Feature Selection을 위한 학습 데이터가 준비되지 않았습니다.")
            return dataset
            
        try:
            print("\n📊 Feature Importance 계산 중...")
            
            # RandomForest 모델 학습
            X_train = dataset.train
            y_train = self.dataset.label
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Feature Importance 계산
            importances = model.feature_importances_
            feature_names = X_train.columns
            
            # Feature Importance 데이터프레임 생성
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            print("\n📌 Feature Importance 상위 10개:")
            print(feature_importance.head(10))
            
            # 중요도가 0.01 미만인 변수 선택
            low_importance_features = feature_importance[feature_importance['Importance'] < 0.01]['Feature'].tolist()
            
            if low_importance_features:
                print(f"\n🗑️ 제거할 변수 (중요도 < 0.01): {low_importance_features}")
                
                # 학습 및 테스트 데이터에서 중요도가 낮은 변수 제거
                dataset.train = dataset.train.drop(columns=low_importance_features)
                dataset.test = dataset.test.drop(columns=low_importance_features)
                print(f"✅ 총 {len(low_importance_features)}개 변수 제거 완료")
            else:
                print("\n✅ 모든 변수가 충분한 중요도를 가집니다 (>= 0.01).")
            
        except Exception as e:
            print(f"⚠️ Feature Selection 중 오류 발생: {e}")
            print("원본 데이터셋을 그대로 사용합니다.")
        
        return dataset

    # 머신러닝 : learning
    def create_k_fold(self, X, y, n_splits=10):
        """
        교차검증을 위한 StratifiedKFold를 생성합니다.
        10-Fold 교차검증을 적용합니다.
        """
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def create_random_variable(self):
        X_train = self.dataset.train
        y_train = self.dataset.label
        X_test = self.dataset.test
        return X_train, y_train, X_test
        
    def evaluate_model(self, model_class, X, y, **kwargs):
        """
        교차검증을 통해 모델의 평균 정확도를 계산합니다.
        
        Args:
            model_class: 모델 클래스 또는 모델 인스턴스
            X: 특성 데이터
            y: 타겟 레이블
            **kwargs: 모델 생성 시 전달할 추가 파라미터
            
        Returns:
            float: 교차검증 평균 정확도
        """
        kf = self.create_k_fold(X, y)
        accuracies = []
        
        for train_index, test_index in kf.split(X, y):
            X_fold_train, X_fold_test = X.iloc[train_index], X.iloc[test_index]
            y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index]
            
            # 모델이 이미 인스턴스인지 확인
            if isinstance(model_class, type):
                model = model_class(**kwargs)
            else:
                model = model_class
                
            model.fit(X_fold_train, y_fold_train)
            
            pred = model.predict(X_fold_test)
            accuracies.append(accuracy_score(y_fold_test, pred))
        
        return np.mean(accuracies)

    def accuracy_by_dtree(self, X, y):
        avg_accuracy = self.evaluate_model(DecisionTreeClassifier, X, y, random_state=42)
        print(f"결정트리 평균 정확도: {avg_accuracy:.4f}")
        return avg_accuracy
    # if-else처럼 데이터를 분할하며 조건 트리를 만듦 (예: 성별 → 나이 → 객실등급).
    # 데이터를 가장 잘 구분할 수 있는 특성부터 if-else처럼 나누며 생존 여부를 예측합니다. 
    # 타이타닉 데이터에서는 성별, 등급, 나이 같은 변수들이 주요 분기 기준으로 사용
    # 훈련데이터에 맞춰 학습을 하다보니 새로운 데이터가 들어오면 예측 정확도가 떨어질 수 있음 (ex. 여성이면 생존으로 예측하는 경우)

    def accuracy_by_random_forest(self, X, y):
        avg_accuracy = self.evaluate_model(RandomForestClassifier, X, y, random_state=42)
        print(f"랜덤포레스트 평균 정확도: {avg_accuracy:.4f}")
        return avg_accuracy
    # 여러 결정트리를 학습한 후 결과를 투표로 종합 (앙상블)

    def accuracy_by_naive_bayes(self, X, y):
        avg_accuracy = self.evaluate_model(GaussianNB, X, y)
        print(f"나이브베이즈 평균 정확도: {avg_accuracy:.4f}")
        return avg_accuracy
    # 각 특성이 독립이라고 가정하고 확률 기반으로 분류

    def accuracy_by_knn(self, X, y):
        avg_accuracy = self.evaluate_model(KNeighborsClassifier, X, y, n_neighbors=5)
        print(f"K-최근접이웃 평균 정확도: {avg_accuracy:.4f}")
        return avg_accuracy
    # 새 데이터가 주변 이웃(K개)과 얼마나 가까운지로 판단

    def accuracy_by_svm(self, X, y):
        avg_accuracy = self.evaluate_model(SVC, X, y, random_state=42)
        print(f"서포트벡터머신 평균 정확도: {avg_accuracy:.4f}")
        return avg_accuracy
    # 데이터를 분리하는 가장 넓은 경계(초평면)를 찾아 분류

    def accuracy_by_voting_ensemble(self, X, y):
        """
        랜덤포레스트, KNN, SVM을 조합한 앙상블 모델(VotingClassifier)의 정확도를 평가합니다.
        """
        # 개별 모델 정의
        rf = RandomForestClassifier(random_state=42)
        knn = KNeighborsClassifier(n_neighbors=5)
        svm = SVC(probability=True, random_state=42)  # soft voting을 위해 probability=True 설정
        
        # VotingClassifier 생성 (soft voting)
        voting_clf = VotingClassifier(
            estimators=[('rf', rf), ('knn', knn), ('svm', svm)],
            voting='soft'
        )
        
        # 교차검증 정확도 계산
        avg_accuracy = self.evaluate_model(voting_clf, X, y)
        print(f"앙상블(Voting) 평균 정확도: {avg_accuracy:.4f}")
        return avg_accuracy
    
    def optimize_random_forest(self, X, y):
        """
        GridSearchCV를 사용하여 랜덤포레스트 모델의 하이퍼파라미터를 최적화합니다.
        """
        print("\n🔍 랜덤포레스트 하이퍼파라미터 최적화 시작...")
        
        # 탐색할 하이퍼파라미터 그리드 정의
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # 기본 랜덤포레스트 모델 생성
        rf = RandomForestClassifier(random_state=42)
        
        # GridSearchCV 설정
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=self.create_k_fold(X, y),
            scoring='accuracy',
            n_jobs=-1,  # 모든 CPU 사용
            verbose=1
        )
        
        try:
            # 그리드 서치 수행
            grid_search.fit(X, y)
            
            # 최적 파라미터 및 점수 출력
            print(f"\n✅ 최적 하이퍼파라미터: {grid_search.best_params_}")
            print(f"🏆 최적 교차검증 정확도: {grid_search.best_score_:.4f}")
            
            # 최적 모델 설정
            best_rf = grid_search.best_estimator_
            
            # 교차검증 정확도 계산
            kf = self.create_k_fold(X, y)
            accuracies = []
            
            for train_index, test_index in kf.split(X, y):
                X_fold_train, X_fold_test = X.iloc[train_index], X.iloc[test_index]
                y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index]
                
                # 이미 학습된 모델이므로 재학습하지 않음
                pred = best_rf.predict(X_fold_test)
                accuracies.append(accuracy_score(y_fold_test, pred))
            
            avg_accuracy = np.mean(accuracies)
            print(f"최적화된 랜덤포레스트 평균 정확도: {avg_accuracy:.4f}")
            
            return avg_accuracy, best_rf
            
        except Exception as e:
            print(f"⚠️ 하이퍼파라미터 최적화 중 오류 발생: {e}")
            print("기본 랜덤포레스트 모델로 대체합니다.")
            return self.accuracy_by_random_forest(X, y), None

    def find_best_model(self):
        print("🔍 최적 모델 선택 시작...")
        X, y, _ = self.create_random_variable()
        
        # 각 모델 평가
        models = {
            "결정트리": self.accuracy_by_dtree,
            "랜덤포레스트": self.accuracy_by_random_forest,
            "나이브베이즈": self.accuracy_by_naive_bayes,
            "K-최근접이웃": self.accuracy_by_knn,
            "서포트벡터머신": self.accuracy_by_svm,
            "XGBoost": self.accuracy_by_xgboost,
            "LightGBM": self.accuracy_by_lightgbm,
            "앙상블(Voting)": self.accuracy_by_voting_ensemble
        }
        
        results = {}
        for name, func in models.items():
            print(f"\n{name} 평가 중...")
            accuracy = func(X, y)
            results[name] = accuracy
        
        # 랜덤포레스트 하이퍼파라미터 최적화
        print("\n랜덤포레스트 하이퍼파라미터 최적화 중...")
        rf_accuracy, best_rf = self.optimize_random_forest(X, y)
        results["최적화된 랜덤포레스트"] = rf_accuracy
        
        # 정확도 기준 내림차순 정렬
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        # 결과 표 형태로 출력
        print("\n📊 모델 정확도 순위:")
        print("=" * 40)
        print(f"{'모델명':<25} {'정확도':>10}")
        print("-" * 40)
        for i, (name, accuracy) in enumerate(sorted_results.items(), 1):
            print(f"{i:2d}. {name:<22} {accuracy:.4f}")
        print("=" * 40)
        
        best_model = list(sorted_results.keys())[0]
        best_accuracy = sorted_results[best_model]
        
        print(f"\n🏆 최고 성능 모델: {best_model} (정확도: {best_accuracy:.4f})")
        return best_model, best_accuracy
    
    def accuracy_by_xgboost(self, X, y):
        """XGBoost 모델의 정확도를 평가합니다."""
        if XGBClassifier is None:
            print("⚠️ XGBoost 라이브러리가 설치되어 있지 않습니다.")
            return 0.0
        
        try:
            # 기본 XGBoost 모델 생성
            avg_accuracy = self.evaluate_model(
                XGBClassifier, X, y, 
                random_state=42,
                use_label_encoder=False,  # 경고 방지
                eval_metric='logloss'     # 경고 방지
            )
            print(f"XGBoost 평균 정확도: {avg_accuracy:.4f}")
            return avg_accuracy
        except Exception as e:
            print(f"⚠️ XGBoost 평가 중 오류 발생: {e}")
            return 0.0
    
    def accuracy_by_lightgbm(self, X, y):
        """LightGBM 모델의 정확도를 평가합니다."""
        if LGBMClassifier is None:
            print("⚠️ LightGBM 라이브러리가 설치되어 있지 않습니다.")
            return 0.0
        
        try:
            # 기본 LightGBM 모델 생성
            avg_accuracy = self.evaluate_model(
                LGBMClassifier, X, y, 
                random_state=42,
                verbose=-1  # 출력 최소화
            )
            print(f"LightGBM 평균 정확도: {avg_accuracy:.4f}")
            return avg_accuracy
        except Exception as e:
            print(f"⚠️ LightGBM 평가 중 오류 발생: {e}")
            return 0.0
    
    def create_voting_ensemble(self):
        """
        랜덤포레스트, KNN, SVM 모델을 조합한 VotingClassifier(앙상블 분류기)를 생성합니다.
        """
        rf = RandomForestClassifier(random_state=42)
        knn = KNeighborsClassifier(n_neighbors=5)
        svm = SVC(probability=True, random_state=42)

        voting_clf = VotingClassifier(
            estimators=[('rf', rf), ('knn', knn), ('svm', svm)],
            voting='soft'
        )
        return voting_clf
    
    
    
    
    