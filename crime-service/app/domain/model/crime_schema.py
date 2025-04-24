from dataclasses import dataclass

@dataclass
class CrimeSchema:
    cctv : object
    crime : object
    pop : object


    @property
    def cctv(self) -> object:
        return self._cctv
    @cctv.setter
    def cctv(self,cctv):
        self._cctv = cctv
    @property
    def crime(self) -> str:
        return self._crime
    @crime.setter
    def crime(self,crime):
        self._crime = crime
    @property
    def pop(self) -> str:
        return self._pop
    @pop.setter
    def pop(self,pop):
        self._pop = pop