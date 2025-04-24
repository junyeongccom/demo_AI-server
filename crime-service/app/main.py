from app.domain.controller.crime_controller import CrimeController

if __name__ == '__main__':
    print("🚨 [Crime] 전처리 시작")

    controller = CrimeController()
    dataset = controller.preprocess()

    print("✅ [Crime] 전처리 완료! 컬럼 확인:")
    print(" - 📊 CCTV:", dataset.cctv.columns.tolist())
    print(" - 📊 Crime:", dataset.crime.columns.tolist())
    print(" - 📊 Pop:", dataset.pop.columns.tolist())