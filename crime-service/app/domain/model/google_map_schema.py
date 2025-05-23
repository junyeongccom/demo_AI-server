import os
from dotenv import load_dotenv
import googlemaps

load_dotenv()

class GoogleMapSchema:
    _instance = None  # 싱글턴 인스턴스를 저장할 클래스 변수

    def __new__(cls):
        if cls._instance is None:  # 인스턴스가 없으면 생성
            cls._instance = super(GoogleMapSchema, cls).__new__(cls)
            cls._instance._api_key = cls._instance._retrieve_api_key()  # API 키 가져오기
            cls._instance._client = googlemaps.Client(key=cls._instance._api_key)  # Google Maps API 클라이언트 초기화
        return cls._instance  # 기존 인스턴스 반환

    def _retrieve_api_key(self):
        return os.getenv("GOOGLE_MAPS_API_KEY")

    def get_api_key(self):
        """저장된 API 키 반환"""
        return self._api_key

    def geocode(self, address, language='ko'):
        """주소를 위도, 경도로 변환하는 메서드"""
        return self._client.geocode(address, language=language)