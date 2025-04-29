from app.domain.controller.matjib_controller import MatjibController

from fastapi import FastAPI

app = FastAPI()

if __name__ == '__main__':
    print("🍽️ [Matjib] 맛집 데이터 전처리 시작")

    controller = MatjibController()
    dataset = controller.preprocess()

    print("✅ [Matjib] 전처리 완료! 컬럼 목록:")
    print(dataset.columns.tolist())