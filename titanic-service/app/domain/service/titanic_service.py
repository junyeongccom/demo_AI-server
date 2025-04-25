import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

    def new_model(self, fname: str) -> pd.DataFrame:
        path = Path(self.dataset.context) / fname
        return pd.read_csv(path)

    def preprocess(self, train_fname: str, test_fname: str) -> TitanicSchema:
        print("-------- 모델 전처리 시작 --------")
        ds = self.dataset
        ds.train = self.new_model(train_fname)
        ds.test = self.new_model(test_fname)
        ds.id = ds.test['PassengerId']
        ds.label = ds.train['Survived']
        ds.train = ds.train.drop('Survived', axis=1)

        ds = self.drop_feature(ds, 'SibSp', 'Parch', 'Cabin', 'Ticket')
        ds = self.extract_title_from_name(ds)
        title_mapping = self.remove_duplicate_title(ds)
        ds = self.title_nominal(ds, title_mapping)
        ds = self.drop_feature(ds, 'Name')
        ds = self.gender_nominal(ds)
        ds = self.drop_feature(ds, 'Sex')
        ds = self.embarked_nominal(ds)
        ds = self.age_ratio(ds)
        ds = self.drop_feature(ds, 'Age')
        ds = self.pclass_ordinal(ds)
        ds = self.fare_ordinal(ds)
        ds = self.drop_feature(ds, 'Fare')

        return ds

    def drop_feature(self, dataset: TitanicSchema, *features: str) -> TitanicSchema:
        for col in features:
            dataset.train.drop(columns=col, inplace=True)
            dataset.test.drop(columns=col, inplace=True)
        return dataset

    def extract_title_from_name(self, dataset: TitanicSchema) -> TitanicSchema:
        for df in [dataset.train, dataset.test]:
            df['Title'] = df['Name'].str.extract('([A-Za-z]+)\\.', expand=False)
        return dataset

    def remove_duplicate_title(self, dataset: TitanicSchema) -> dict:
        all_titles = set(dataset.train['Title'].unique()) | set(dataset.test['Title'].unique())
        print(f"📌 전체 직함 목록: {sorted(all_titles)}")
        return {
            'Mr': 1, 'Ms': 2, 'Mrs': 3, 'Master': 4,
            'Royal': 5, 'Rare': 6
        }

    def title_nominal(self, dataset: TitanicSchema, title_mapping: dict) -> TitanicSchema:
        for df in [dataset.train, dataset.test]:
            df['Title'] = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            df['Title'] = df['Title'].replace(
                ['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Mme'], 'Rare')
            df['Title'] = df['Title'].replace({'Mlle': 'Mr', 'Miss': 'Ms'})
            df['Title'] = df['Title'].map(title_mapping)
        return dataset

    def gender_nominal(self, dataset: TitanicSchema) -> TitanicSchema:
        mapping = {'male': 1, 'female': 2}
        for df in [dataset.train, dataset.test]:
            df['Gender'] = df['Sex'].map(mapping)
        return dataset

    def embarked_nominal(self, dataset: TitanicSchema) -> TitanicSchema:
        mapping = {'S': 1, 'C': 2, 'Q': 3}
        for df in [dataset.train, dataset.test]:
            df['Embarked'] = df['Embarked'].fillna('S')
            df['Embarked'] = df['Embarked'].map(mapping)
        return dataset

    def age_ratio(self, dataset: TitanicSchema) -> TitanicSchema:
        mapping = {'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4,
                   'Young Adult': 5, 'Adult': 6, 'Senior': 7}
        for df in [dataset.train, dataset.test]:
            df['Age'] = df['Age'].fillna(-1)

        max_age = max(dataset.train['Age'].max(), dataset.test['Age'].max())
        bins = [-1] + [max_age * i / 7 for i in range(1, 8)]
        labels = list(mapping.keys())[1:]  # 'Baby' ~ 'Senior'

        for df in [dataset.train, dataset.test]:
            df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
            df['AgeGroup'] = df['AgeGroup'].map(mapping)
        return dataset

    def fare_ordinal(self, dataset: TitanicSchema) -> TitanicSchema:
        mapping = {'Free': 1, 'Lower': 2, 'Mid': 3, 'Premium': 4}
        bins = [0, 0.01, 20, 150, 870]
        labels = list(mapping.keys())

        for df in [dataset.train, dataset.test]:
            df['Fare'] = df['Fare'].fillna(0)
            df['FareGroup'] = pd.cut(df['Fare'], bins=bins, labels=labels, include_lowest=True)
            df['FareGroup'] = df['FareGroup'].map(mapping)
        return dataset

    def pclass_ordinal(self, dataset: TitanicSchema) -> TitanicSchema:
        return dataset

    # 머신러닝 : learning
    def create_k_fold(self, X, y, n_splits=5):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def create_random_variable(self):
        X_train = self.dataset.train
        y_train = self.dataset.label
        X_test = self.dataset.test
        return X_train, y_train, X_test
        
    def evaluate_model(self, model_class, X, y, **kwargs):
        """
        모델을 평가하는 공통 함수
        
        Args:
            model_class: 사용할 모델 클래스 (예: DecisionTreeClassifier)
            X: 입력 데이터
            y: 타겟 레이블
            **kwargs: 모델 클래스에 전달할 추가 매개변수
            
        Returns:
            float: 평균 정확도
        """
        kf = self.create_k_fold(X, y)
        accuracies = []
        
        for train_index, test_index in kf.split(X, y):
            X_fold_train, X_fold_test = X.iloc[train_index], X.iloc[test_index]
            y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index]
            
            model = model_class(**kwargs)
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

    def find_best_model(self):
        print("🔍 최적 모델 선택 시작...")
        X, y, _ = self.create_random_variable()
        
        # 각 모델 평가
        models = {
            "결정트리": self.accuracy_by_dtree,
            "랜덤포레스트": self.accuracy_by_random_forest,
            "나이브베이즈": self.accuracy_by_naive_bayes,
            "K-최근접이웃": self.accuracy_by_knn,
            "서포트벡터머신": self.accuracy_by_svm
        }
        
        results = {}
        for name, func in models.items():
            print(f"\n{name} 평가 중...")
            accuracy = func(X, y)
            results[name] = accuracy
        
        # 정확도 기준 내림차순 정렬
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        print("\n📊 모델 정확도 순위:")
        for i, (name, accuracy) in enumerate(sorted_results.items(), 1):
            print(f"{i}위: {name} - {accuracy:.4f}")
        
        best_model = list(sorted_results.keys())[0]
        best_accuracy = sorted_results[best_model]
        
        print(f"\n🏆 최고 성능 모델: {best_model} (정확도: {best_accuracy:.4f})")
        return best_model, best_accuracy
    
    
    
    
    