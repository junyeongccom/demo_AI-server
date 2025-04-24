# 머신러닝 모델 API 서버

FastAPI를 사용한 머신러닝 모델 배포 API 서버입니다.

## 프로젝트 구조

```
📁 app/
├── 📁 controller/         ← API 라우터 정의
│   └── predict_controller.py
├── 📁 service/            ← 비즈니스 로직
│   └── predict_service.py
├── main.py               ← FastAPI 앱 실행
├── model.pkl             ← 저장된 ML 모델
📄 Dockerfile              ← 도커 이미지 정의
📄 requirements.txt        ← 의존 패키지 리스트
📄 .dockerignore           ← 도커 빌드시 제외할 파일 목록
📄 README.md               ← 프로젝트 설명서
```

## 설치 및 실행 방법

### 로컬 실행

1. 필요한 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

2. 서버 실행:
   ```
   uvicorn app.main:app --reload
   ```

3. API 문서 확인:
   ```
   http://localhost:8000/docs
   ```

### Docker 실행

1. Docker 이미지 빌드:
   ```
   docker build -t ml-api-server .
   ```

2. Docker 컨테이너 실행:
   ```
   docker run -p 8000:8000 ml-api-server
   ```

## API 사용 방법

### 예측 API

- 엔드포인트: `/predict`
- 메소드: `POST`
- 요청 형식:
  ```json
  {
    "features": {
      "feature1": 값1,
      "feature2": 값2,
      ...
    }
  }
  ```
- 응답 형식:
  ```json
  {
    "prediction": 예측값
  }
  ```

## 모델 교체 방법

`app/model.pkl` 파일을 자신의 학습된 모델로 교체하세요. 모델은 scikit-learn의 `pickle.dump()` 메소드로 저장된 것이어야 합니다.

예시:
```python
import pickle
from sklearn.ensemble import RandomForestClassifier

# 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 모델 저장
with open('app/model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

## 커스터마이징

1. 특성 처리 로직을 수정하려면 `app/service/predict_service.py` 파일의 `predict` 메소드를 수정하세요.
2. 추가 엔드포인트를 만들려면 `app/controller/` 디렉토리에 새 컨트롤러를 추가하고 `app/main.py`에 라우터를 등록하세요. 