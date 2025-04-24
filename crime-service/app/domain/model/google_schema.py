import googlemaps


class ApiKeyManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ApiKeyManager, cls).__new__(cls)
            cls._instance._api_key = cls._instance._set_api_key()
            cls._instance._client = googlemaps.Client(key=cls._instance._api_key)  # Google Maps API 클라이언트 초기화
        return cls._instance

    def _set_api_key(self):
        return "..." #API 키값 넣는 곳

    def get_api_key(self):
        return self._api_key
    
    def geocode(self, address, language='ko'):
        """주소를 위도, 경도로 변환하는 메서드"""
        return self._client.geocode(address, language=language)