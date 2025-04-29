from pydantic import BaseModel
from typing import Optional

# ✅ 타이타닉 요청 모델 정의
class TitanicRequest(BaseModel):
    # 타이타닉 서비스 요청에 필요한 필드들
    pclass: Optional[int] = None
    name: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[float] = None
    sibsp: Optional[int] = None
    parch: Optional[int] = None
    ticket: Optional[str] = None
    fare: Optional[float] = None
    cabin: Optional[str] = None
    embarked: Optional[str] = None 