from app.domain.controller.titanic_controller import TitanicController

if __name__ == '__main__':
    print("🚢 [Titanic] 타이타닉 데이터 전처리 시작")

    controller = TitanicController()
    dataset = controller.preprocess()

    print("✅ [Titanic] 전처리 완료! 컬럼 목록:")
    print(dataset.train.columns.tolist())
