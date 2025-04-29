# app/domain/model/service_type.py

import os
from enum import Enum

class ServiceType(str, Enum):
    TITANIC = "titanic"
    CRIME = "crime"
    MATJIB = "matjib"

# 서비스별 Base URL 매핑
SERVICE_URLS = {
    ServiceType.TITANIC: os.getenv("TITANIC_SERVICE_URL"),
    ServiceType.CRIME: os.getenv("CRIME_SERVICE_URL"),
    ServiceType.MATJIB: os.getenv("MATJIB_SERVICE_URL"),
}
