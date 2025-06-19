# 장소 추천 및 경로 시각화 시스템

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
| `start.ipynb` | 사용자에게 MBTI, 계절, 위치 정보를 입력받고 추천 결과를 실행하는 메인 노트북입니다. |
| `project.ipynb` | 시각화, 클러스터링 테스트 등 개발 및 실험용 노트북입니다. |
| `process_recommend_place.py` | 사용자 입력을 받아 추천 → 경로 계산 → 지도 생성까지 수행하는 주요 로직입니다. |
| `train_model.py` | 장소 특성 데이터를 UMAP + KMeans + DBSCAN 등으로 클러스터링하고 모델을 학습·저장합니다. |
| `request_weather.py` | 행정구역 위치 기반으로 날씨 API를 비동기 요청하여 날씨 데이터를 수집합니다. |
| `kakao_route_service.py` | Kakao API 및 OpenRouteService를 사용하여 좌표 변환, 거리 계산, 경로 시각화, 주변 장소 탐색을 담당합니다. |
| `README.md` | 프로젝트 개요, 설치 방법, 실행 예시 등을 설명한 문서입니다. |
| `requirements.txt` | 필요한 Python 라이브러리 목록입니다. `pip install -r requirements.txt`로 설치합니다. |
| `.env` | 민감한 API 키 (Kakao, OpenRouteService, Weather API 등)를 보관하는 환경 설정 파일입니다. GitHub에 업로드하지 않도록 주의합니다. |

### data/

| 파일명 | 설명 |
|--------|------|
| `place3_3.csv` | 장소 기본 정보, 계절별 태그, 카테고리 등이 포함된 원본 장소 데이터입니다. |
| `location(weather).csv` | 행정구역 코드별로 격자 X, Y 위치와 위도/경도 정보가 포함된 날씨용 보조 데이터입니다. |

### models/

| 파일명 | 설명 |
|--------|------|
| `train_model.py` 실행 후 생성된 폴더입니다. |
| `X_reduced.npy` | UMAP으로 차원 축소된 장소 특성 벡터입니다. |
| `clustered_places.csv` | 클러스터링 결과가 포함된 장소 정보입니다 (원본 + 예측 클러스터 컬럼 포함). |
| `scaler.pkl` | 학습 시 사용된 `StandardScaler` 객체입니다. |
| `reducer.pkl` | 학습된 `UMAP` 차원 축소 모델입니다. |
| `kmeans.pkl` | KMeans 클러스터링 모델입니다. |
| `dbscan.pkl` | DBSCAN 클러스터링 모델입니다. |
| `agg_cluster.pkl` | 계층적 군집화(Agglomerative Clustering) 모델입니다. |
| `tsne.pkl` | t-SNE 기반 시각화용 모델 객체입니다. |


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


