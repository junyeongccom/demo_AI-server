from dataclasses import dataclass

@dataclass
class MatjibSchema:
    matjib: object
    context: str
    fname: int
    id: str
    name: str

    @property
    def matjib(self) -> object:
        return self._matjib

    @matjib.setter
    def matjib(self, matjib):
        self._matjib = matjib

    @property
    def context(self) -> str:
        return self._context

    @context.setter
    def context(self, context):
        self._context = context

    @property
    def fname(self) -> str:
        return self._fname

    @fname.setter
    def fname(self, fname):
        self._fname = fname

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name