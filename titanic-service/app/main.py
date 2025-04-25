from app.domain.controller.titanic_controller import TitanicController

if __name__ == '__main__':
    print("🚢 [Titanic] 타이타닉 데이터 전처리 시작")

    controller = TitanicController()
    dataset = controller.preprocess()

    print("✅ [Titanic] 전처리 완료! 컬럼 목록:")
    print(dataset.train.columns.tolist())
    
    print("\n🤖 [Titanic] 머신러닝 모델 학습 및 평가 시작")
    best_model, best_accuracy = controller.find_best_model()
    print(f"✅ [Titanic] 모델링 완료! 최적 모델: {best_model}, 정확도: {best_accuracy:.4f}")
