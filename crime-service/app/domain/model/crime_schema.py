from dataclasses import dataclass
from typing import Dict, Any
from pydantic import BaseModel

@dataclass
class CrimeSchema:
    cctv : object
    crime : object
    pop : object
    police : object


    @property
    def cctv(self) -> object:
        return self._cctv
    @cctv.setter
    def cctv(self,cctv):
        self._cctv = cctv
    @property
    def crime(self) -> object:
        return self._crime
    @crime.setter
    def crime(self,crime):
        self._crime = crime
    @property
    def pop(self) -> object:
        return self._pop
    @pop.setter
    def pop(self,pop):
        self._pop = pop
    @property
    def police(self) -> object:
        return self._police
    @police.setter
    def police(self,police):
        self._police = police

# API 응답 모델
class AnalysisResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any] = None