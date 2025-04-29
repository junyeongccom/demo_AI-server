# app/domain/model/service_proxy_factory.py

from typing import Optional
from fastapi import HTTPException
import httpx
from app.domain.model.service_type import ServiceType, SERVICE_URLS

class ServiceProxyFactory:
    def __init__(self, service_type: ServiceType):
        self.base_url = SERVICE_URLS.get(service_type)
        if not self.base_url:
            raise HTTPException(status_code=500, detail=f"서비스 URL을 찾을 수 없습니다: {service_type}")
        
    async def request(
        self,
        method: str,
        path: str,
        headers: list[tuple[bytes, bytes]],
        body: Optional[bytes] = None
    ) -> httpx.Response:
        url = f"{self.base_url}/{path}"

        # 수신한 headers를 딕셔너리로 변환
        headers_dict = {k.decode(): v.decode() for k, v in headers}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers_dict,
                    content=body
                )
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"서비스 요청 실패: {str(e)}")
