from app.domain.service.titanic_service import TitanicService
from app.domain.model.titanic_schema import TitanicSchema
import pandas as pd
import os
from pathlib import Path

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
        """
        모델 학습 및 평가 진행을 위한 간단한 래퍼 메서드
        evaluation()만 호출하여 결과 반환
        """
        print("🧠 머신러닝 모델 학습 및 평가 호출")
        return self.evaluation()

    def evaluation(self):
        """
        여러 머신러닝 모델을 학습하고 교차검증하여 정확도를 비교 평가
        """
        print("🔍 모델 평가 시작...")
        
        # 전처리된 데이터 가져오기
        X, y, X_test = self.titanic_service.create_random_variable()
        
        # X가 None인지 확인
        if X is None or y is None:
            print("⚠️ 데이터 준비 중 오류가 발생했습니다.")
            return None, None, 0.0
        
        # Feature Selection 적용
        print("\n🔍 Feature Selection을 통한 변수 최적화 시작...")
        print("중요도가 낮은 피처(< 0.01)를 제거합니다.")
        X, X_test, removed_features = self.apply_feature_selection(X, y, X_test)
        
        # Feature Selection 후 X가 None인지 다시 확인
        if X is None:
            print("⚠️ Feature Selection 중 오류가 발생했습니다.")
            return None, None, 0.0
        
        if removed_features:
            print(f"\n🗑️ 제거된 피처 목록: {removed_features}")
        else:
            print("\n✅ 모든 피처가 중요도 기준을 통과했습니다.")
        
        # Feature Selection 후 남은 피처 확인
        print(f"\n🔍 Feature Selection 후 남은 피처 목록 ({len(X.columns)}개):")
        print(X.columns.tolist())
        
        # 각 알고리즘별 정확도 계산
        model_results = {}
        print("\n각 모델 평가 중...")
        
        model_results["결정트리"] = self.titanic_service.accuracy_by_dtree(X, y)
        model_results["랜덤포레스트"] = self.titanic_service.accuracy_by_random_forest(X, y) 
        model_results["나이브베이즈"] = self.titanic_service.accuracy_by_naive_bayes(X, y)
        model_results["K-최근접이웃"] = self.titanic_service.accuracy_by_knn(X, y)
        model_results["서포트벡터머신"] = self.titanic_service.accuracy_by_svm(X, y)
        model_results["XGBoost"] = self.titanic_service.accuracy_by_xgboost(X, y)
        model_results["LightGBM"] = self.titanic_service.accuracy_by_lightgbm(X, y)
        model_results["앙상블(Voting)"] = self.titanic_service.accuracy_by_voting_ensemble(X, y)
        
        # 랜덤포레스트 하이퍼파라미터 최적화
        print("\n랜덤포레스트 하이퍼파라미터 최적화 중...")
        rf_accuracy, best_rf = self.titanic_service.optimize_random_forest(X, y)
        model_results["최적화된 랜덤포레스트"] = rf_accuracy
        
        # 정확도 기준 내림차순 정렬
        sorted_results = dict(sorted(model_results.items(), key=lambda x: x[1], reverse=True))
        
        # 결과 표 형태로 출력
        print("\n📊 모델 정확도 순위:")
        print("=" * 40)
        print(f"{'모델명':<25} {'정확도':>10}")
        print("-" * 40)
        for i, (name, accuracy) in enumerate(sorted_results.items(), 1):
            print(f"{i:2d}. {name:<22} {accuracy:.4f}")
        print("=" * 40)
        
        # 최고 성능 모델 선택
        best_model = list(sorted_results.keys())[0]
        best_accuracy = sorted_results[best_model]
        
        print(f"\n🏆 최고 성능 모델: {best_model} (정확도: {best_accuracy:.4f})")
        
        # 이 두 값을 self에 저장하여 다른 메서드에서도 사용할 수 있게 함
        self.selected_features = X.columns.tolist()
        self.removed_features = removed_features
        
        return sorted_results, best_model, best_accuracy
    
    def apply_feature_selection(self, X, y, X_test):
        """
        Feature Importance를 계산하고 중요도가 낮은 피처를 제거합니다.
        
        Args:
            X: 학습 데이터
            y: 타겟 레이블
            X_test: 테스트 데이터
            
        Returns:
            tuple: (변환된 X, 변환된 X_test, 제거된 피처 목록)
        """
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        
        try:
            # RandomForest 모델 학습
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Feature Importance 계산
            importances = model.feature_importances_
            feature_names = X.columns
            
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
                # 중요도가 낮은 변수 제거
                X_filtered = X.drop(columns=low_importance_features)
                X_test_filtered = X_test.drop(columns=low_importance_features)
                print(f"✅ 총 {len(low_importance_features)}개 변수 제거 완료")
                return X_filtered, X_test_filtered, low_importance_features
            else:
                print("✅ 모든 변수가 충분한 중요도를 가집니다 (>= 0.01)")
                return X, X_test, []
        
        except Exception as e:
            print(f"⚠️ Feature Selection 중 오류 발생: {e}")
            print("원본 데이터를 그대로 사용합니다.")
            return X, X_test, []

    def submit(self):
        """
        테스트 데이터에 대한 예측을 수행하고 submission 파일을 생성합니다.
        Feature Selection을 통해 선택된 특성만 사용합니다.
        """
        try:
            # 전처리된 데이터 가져오기
            X, y, X_test = self.titanic_service.create_random_variable()
            
            # Feature Selection 결과가 있는지 확인
            if hasattr(self, 'selected_features') and self.selected_features:
                print(f"\n🔍 Feature Selection 결과를 적용하여 {len(self.selected_features)}개의 피처만 사용합니다.")
                X = X[self.selected_features]
                X_test = X_test[self.selected_features]
                
                if hasattr(self, 'removed_features') and self.removed_features:
                    print(f"🗑️ 제외된 피처: {self.removed_features}")
            else:
                print("\n⚠️ Feature Selection 결과가 없습니다. 모든 피처를 사용합니다.")
                
            print("\n📊 예측 모델(앙상블 보팅) 학습 중...")
            
            # 앙상블 보팅 분류기로 변경 (최고 성능 보장)
            model = self.titanic_service.create_voting_ensemble()
            model.fit(X, y)
            
            print("✅ 모델 학습 완료")
            
            # 테스트 데이터에 대한 예측 수행
            predictions = model.predict(X_test)
            
            # submission 파일 생성
            submission = pd.DataFrame({
                'PassengerId': self.titanic_service.test_ids,
                'Survived': predictions.astype(int)
            })
            
            # submission 파일 저장
            submission_path = 'app/updated_data/submission.csv'
            submission.to_csv(submission_path, index=False)
            
            print(f"\n📄 예측 결과가 {submission_path}에 저장되었습니다.")
            print(f"예측 결과 요약: 생존={sum(predictions)}, 사망={len(predictions)-sum(predictions)}")
            
            return submission
        
        except Exception as e:
            print(f"⚠️ 제출 파일 생성 중 오류 발생: {e}")
            return None

