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
| `data/location(weather).csv` | 행정구역별 위도·경도·격자 정보 데이터 (기상청 API 사용을 위한 좌표 변환용) |
| `data/place3_3.csv` | 장소의 지역명, 카테고리, 계절 정보 등을 담은 원본 입력 데이터셋 |
| `models/clustered_places.csv` | `train_model.py` 실행 결과로 생성된 추천용 클러스터링 결과 데이터 |
| `models/X_reduced.npy` | UMAP을 통해 차원 축소된 벡터 데이터 (후속 시각화, 추천 모델에 사용) |
| `models/kmeans.pkl` | KMeans 클러스터링 모델 객체 |
| `models/agg_cluster.pkl` | 계층적 군집화 모델 객체 |
| `models/dbscan.pkl` | DBSCAN 클러스터링 모델 객체 |
| `models/reducer.pkl` | 학습된 UMAP 차원 축소기 객체 |
| `models/scaler.pkl` | 특성 정규화를 위한 StandardScaler 객체 |
| `models/tsne.pkl` | t-SNE 차원 축소기 객체 (2D/3D 시각화용) |
| `.env` | Kakao, ORS(OpenRouteService), 기상청 API 키 등 환경 변수 설정 파일 |
| `train_model.py` | 클러스터링 학습 파이프라인 전체 수행: UMAP + KMeans/DBSCAN + 결과 저장 |
| `process_recommend_place.py` | MBTI + 계절 + 날씨 + 거리 기반 맞춤형 장소 추천 + 경로 탐색 기능 |
| `kakao_route_service.py` | Kakao 및 ORS API를 활용한 좌표 변환, 거리 계산, 지도 시각화 기능 |
| `request_weather.py` | 기상청 API를 활용해 위치 기반 날씨 정보 수집 비동기 처리 모듈 |
| `start.ipynb` | 사용자 입력(MBTI/계절/주소) 기반 추천 실행 진입점 (최종 실행용) |
| `project.ipynb` | 전체 학습·추천·시각화 기능의 통합 실행 및 실험 기록 노트북 |
| `README.md` | 프로젝트 소개, 실행 방법, 데이터/파일 설명 등을 포함한 문서 |

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
Python==3.12.3
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



