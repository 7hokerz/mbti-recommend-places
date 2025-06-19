## 장소 추천 및 경로 시각화 시스템

이 프로젝트는 사용자의 MBTI, 계절, 주소 정보를 바탕으로 장소를 추천하고, OpenRouteService를 활용해 추천 경로를 지도에 시각화하는 Python 기반 도구입니다. Kakao API를 이용해 주변 장소(예: 카페, 음식점)를 함께 탐색할 수 있습니다.

---

## 주요 기능

- **MBTI + 계절 기반 장소 추천** (`clustered_places.csv` 기반)
- **도로명 주소 → 위경도 좌표 변환** (Kakao API)
- **자동차 최단 경로 계산 및 지도 시각화** (OpenRouteService + Folium)
- **경로 주변 장소(카페, 음식점 등) 마커 표시** (Kakao API)

---

## 설치 방법

1. 이 저장소를 클론합니다.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. 필수 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

3. `.env` 파일을 생성하고 아래와 같이 API 키를 입력합니다.

```env
KAKAO_API_KEY=카카오_REST_API_KEY
ORS_API_KEY=OpenRouteService_API_KEY
WEATHER_API_KEY=기상청_인증키
```

---

## 주요 파일 설명

| 파일명 | 설명 |
|--------|------|
| `start.ipynb` | 사용자에게 MBTI, 계절, 현재 위치(주소)를 입력받아 추천 경로 시각화를 실행하는 노트북 파일입니다. |
| `process_recommend_place.py` | 장소 추천 → 경로 탐색 → 지도 시각화를 순차적으로 수행하는 메인 로직이 포함되어 있습니다. |
| `recommend_places.py` | MBTI + 계절 기반으로 장소를 추천하는 알고리즘을 구현합니다. |
| `kakao_route_service.py` | Kakao API 및 OpenRouteService를 활용하여 좌표 검색, 경로 계산, 지도 생성, 주변 장소 탐색을 수행합니다. |
| `weather_api.py` | 기상청 API를 통해 현재 위치의 날씨 정보를 가져옵니다. |
| `clustered_places.csv` | 장소별 특징(계절/테마) 및 클러스터 정보를 포함한 주요 데이터셋입니다. |
| `.env` | 민감한 API 키를 보관하는 파일로, GitHub에 업로드되지 않도록 `.gitignore`에 포함되어야 합니다. |
| `requirements.txt` | 프로젝트 실행에 필요한 Python 패키지 목록이 정의된 파일입니다. |
| `README.md` | 프로젝트 개요, 설치 방법, 실행 예시 등이 정리된 문서입니다. |

---

## 실행 예시 (Python)

```python
from kakao_route_service import get_driving_distance, create_map_kakao, get_coordinates_unified, get_route_coordinates, search_places

start = "서울시청"
end = "강남역"

# 거리 계산
distance_km = get_driving_distance(start, end)
print(f"{start} → {end}까지 도로 기준 거리: {distance_km}km")

# 경로 및 지도 생성
route_coords = get_route_coordinates(
    get_coordinates_unified(start, is_address=True),
    get_coordinates_unified(end, is_address=False)
)

nearby = search_places(category="CE7", lon=route_coords[1][0], lat=route_coords[1][1])  # 카페
m = create_map_kakao(get_coordinates_unified(start), get_coordinates_unified(end), route_coords, nearby)
m.save("route_map.html")
```

---

## 오픈소스 라이선스

이 프로젝트는 다음과 같은 오픈소스 및 외부 API를 활용하고 있으며, 각 구성 요소는 해당 라이선스를 따릅니다.

| 구성 요소 | 설명 | 라이선스 |
|-----------|------|----------|
| [Kakao Maps API](https://developers.kakao.com/) | 장소 검색, 좌표 변환, 주변 장소 탐색 | Kakao Developers 이용약관 |
| [OpenRouteService](https://openrouteservice.org/) | 도로 기반 경로 탐색 | GNU GPL v3.0 |
| [Folium](https://python-visualization.github.io/folium/) | 지도 시각화 | MIT License |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | 환경 변수 관리 | BSD License |
| [Requests](https://requests.readthedocs.io/) | HTTP 통신 처리 | Apache License 2.0 |
| [Pandas](https://pandas.pydata.org/) | 데이터프레임 처리 | BSD License |

---

## 실험 환경
Python 3.12.7
aiohttp==3.12.7
folium==0.19.7
imbalanced-learn==0.13.0
joblib==1.4.2
matplotlib==3.10.1
nest-asyncio==1.6.0
numpy==1.26.4
openrouteservice==2.3.3
pandas==2.2.3
python-dotenv==1.1.0
requests==2.32.3
scikit-learn==1.6.1
seaborn==0.13.2
tabulate==0.9.0
umap-learn==0.5.7



