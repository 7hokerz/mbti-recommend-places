import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import request_weather
import asyncio
from Server_v1 import get_coordinates_unified, get_driving_distance, create_map_kakao, search_places, get_route_coordinates
import webbrowser

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 모델 디렉토리 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

 # mbti별 각 카테고리에 대해서 가중치 설정
MBTI_WEIGHTS = {
    "ISTJ": {"레저/스포츠": 0.5, "전통/역사": 1.0, "감성/체험거리": 0, "조망/전망": 0.5, "자연물": 0.5, "문화시설감상": 0.5},
    "ISFJ": {"레저/스포츠": 0, "전통/역사": 1.0, "감성/체험거리": 1.0, "조망/전망": 1.0, "자연물": 1.0, "문화시설감상": 0.5},
    "INFJ": {"레저/스포츠": 0, "전통/역사": 0.5, "감성/체험거리": 1.0, "조망/전망": 1.0, "자연물": 1.0, "문화시설감상": 1.0},
    "INTJ": {"레저/스포츠": 0, "전통/역사": 1.0, "감성/체험거리": 0, "조망/전망": 0.5, "자연물": 0.5, "문화시설감상": 1.0},
    "ISTP": {"레저/스포츠": 1.0, "전통/역사": 0, "감성/체험거리": 0, "조망/전망": 0.5, "자연물": 1.0, "문화시설감상": 0},
    "ISFP": {"레저/스포츠": 1.0, "전통/역사": 0.5, "감성/체험거리": 1.0, "조망/전망": 1.0, "자연물": 1.0, "문화시설감상": 0.5},
    "INFP": {"레저/스포츠": 0, "전통/역사": 0.5, "감성/체험거리": 1.0, "조망/전망": 1.0, "자연물": 1.0, "문화시설감상": 1.0},
    "INTP": {"레저/스포츠": 0, "전통/역사": 1.0, "감성/체험거리": 0, "조망/전망": 0.5, "자연물": 0.5, "문화시설감상": 1.0},
    "ESTP": {"레저/스포츠": 1.0, "전통/역사": 0, "감성/체험거리": 0.5, "조망/전망": 0, "자연물": 0.5, "문화시설감상": 0},
    "ESFP": {"레저/스포츠": 1.0, "전통/역사": 0, "감성/체험거리": 1.0, "조망/전망": 1.0, "자연물": 0.5, "문화시설감상": 0},
    "ENFP": {"레저/스포츠": 1.0, "전통/역사": 0.5, "감성/체험거리": 1.0, "조망/전망": 0.5, "자연물": 1.0, "문화시설감상": 1.0},
    "ENTP": {"레저/스포츠": 1.0, "전통/역사": 0.5, "감성/체험거리": 0.5, "조망/전망": 0, "자연물": 0.5, "문화시설감상": 1.0},
    "ESTJ": {"레저/스포츠": 0.5, "전통/역사": 1.0, "감성/체험거리": 0, "조망/전망": 0.5, "자연물": 0.5, "문화시설감상": 0.5},
    "ESFJ": {"레저/스포츠": 0.5, "전통/역사": 1.0, "감성/체험거리": 1.0, "조망/전망": 0.5, "자연물": 0.5, "문화시설감상": 0.5},
    "ENFJ": {"레저/스포츠": 0.5, "전통/역사": 1.0, "감성/체험거리": 1.0, "조망/전망": 0.5, "자연물": 0.5, "문화시설감상": 1.0},
    "ENTJ": {"레저/스포츠": 1.0, "전통/역사": 1.0, "감성/체험거리": 0, "조망/전망": 0.5, "자연물": 0.5, "문화시설감상": 0.5},
}

# 날씨에 따른 가중치 설정
WEATHER_WEIGHTS = {
    '맑음': 1.5,
    '구름많음': 0.5,
    '흐림': -1.0,
    '없음': 0.2, # 강수없음
    '비': -2.0,
    '비/눈': -2.5,
    '눈': -2.0,
    '소나기': -1.5
}

# 모델, 스케일러, 차원 축소, 클러스터링 불러오기
try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    reducer = joblib.load(os.path.join(MODEL_DIR, 'reducer.pkl'))
    kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
    tsne = joblib.load(os.path.join(MODEL_DIR, 'tsne.pkl'))
    X_reduced = np.load(os.path.join(MODEL_DIR, 'X_reduced.npy'))
    place_features = pd.read_csv(os.path.join(MODEL_DIR, 'clustered_places.csv'))
except FileNotFoundError:
    print("오류: 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
    exit()

# 장소 추천 함수
async def recommend_places(mbti: str, 계절: str, 현재주소: str):
    get_weight = MBTI_WEIGHTS.get(mbti.upper())
    if not get_weight:
        raise ValueError(f"MBTI 유형 '{mbti}'를 찾을 수 없습니다.")

    # 개별 feature 값
    레저_스포츠 = get_weight["레저/스포츠"]
    전통_역사 = get_weight["전통/역사"]
    감성_체험거리 = get_weight["감성/체험거리"]
    조망_전망 = get_weight["조망/전망"]
    자연물 = get_weight["자연물"]
    문화시설감상 = get_weight["문화시설감상"]

    # 파생 변수
    자연_조망_결합 = 자연물 * 조망_전망
    자연_역사_결합 = 자연물 * 전통_역사
    자연_레저_결합 = 자연물 * 레저_스포츠
    자연_체험_결합 = 자연물 * 감성_체험거리

    조망_체험_결합 = 조망_전망 * 감성_체험거리
    조망_역사_결합 = 조망_전망 * 전통_역사

    역사_문화_결합 = 전통_역사 * 문화시설감상
    레저_체험_결합 = 레저_스포츠 * 감성_체험거리

    # 최종 feature 벡터
    user_features_data = [[
        레저_스포츠, 전통_역사, 감성_체험거리, 조망_전망, 자연물, 문화시설감상,
        자연_조망_결합, 자연_역사_결합, 자연_레저_결합, 자연_체험_결합, 
        역사_문화_결합, 레저_체험_결합, 조망_체험_결합, 조망_역사_결합
    ]]
    user_features_df = pd.DataFrame(user_features_data, columns=scaler.feature_names_in_)

    # 스케일링 > UMAP > 모델 예측
    user_scaled = scaler.transform(user_features_df)
    user_reduced = reducer.transform(user_scaled)
    user_umap_point = user_reduced[0]  # 사용자의 UMAP 공간 좌표

    # 클러스터 중심점과 거리 계산
    cluster_centroids = kmeans.cluster_centers_
    distances_to_centroids = np.sqrt(np.sum((cluster_centroids - user_umap_point)**2, axis=1))

    N_CLOSEST_CLUSTERS = 3  # 예: 가장 가까운 클러스터 3개 선택
    K_ITEMS_PER_CLUSTER = 5 # 예: 각 클러스터에서 상위 5개 장소 선택

    closest_cluster_labels = np.argsort(distances_to_centroids)[:N_CLOSEST_CLUSTERS]
    
    user_lon, user_lat = get_coordinates_unified(현재주소)
    if user_lon is None or user_lat is None:
        raise ValueError(f"주소 '{현재주소}'의 좌표를 찾을 수 없습니다.")
    
    recommended_places_list = []
    # 거리 계산 및 장소 선택
    for cluster_label in closest_cluster_labels:
        df_current_cluster_season = place_features[
            (place_features['cluster'] == cluster_label) &
            (place_features[계절] == 1) ].copy()
        
        if df_current_cluster_season.empty:
            continue

        # 거리 계산
        indices_in_X_reduced = df_current_cluster_season.index

        points_for_distance_calc = X_reduced[indices_in_X_reduced]
        
        distances_per_place = np.sqrt(np.sum((points_for_distance_calc - user_umap_point)**2, axis=1))

        df_current_cluster_season['distance'] = distances_per_place

        top_k_in_cluster = df_current_cluster_season.sort_values('distance').head(K_ITEMS_PER_CLUSTER)
        recommended_places_list.append(top_k_in_cluster)
        
    # 클러스터별 추천 장소 합침
    final_recommendations_df = pd.concat(recommended_places_list).sort_values('distance')

    # 실제 거리 계산 (수정한 부분)
    final_recommendations_df['real_distance_km'] = final_recommendations_df['ITS_BRO_NM'].apply(
    lambda place: get_driving_distance(현재주소, place)
    )

    # 실시간 날씨 받아오기
    weather_results = await request_weather.get_weather_for_dataframe_async(final_recommendations_df)

    # 날씨 데이터 병합
    df_current = pd.DataFrame(pd.Series(weather_results), columns=['data_dict'])

    df_expanded = df_current['data_dict'].apply(pd.Series)

    df_expanded.index = final_recommendations_df.index

    df_merged = final_recommendations_df.join(df_expanded)

    # 날씨 점수 계산
    weather_cols = [col for col in WEATHER_WEIGHTS.keys() if col in df_merged.columns]
    df_merged['WeatherScore'] = sum(df_merged[col] * WEATHER_WEIGHTS[col] for col in weather_cols)

    scaler2 = MinMaxScaler()

    # 거리는 값이 여러 개일 때만 정규화 수행
    if len(df_merged['distance'].unique()) > 1:
        df_merged['distance_score'] = 1 - scaler2.fit_transform(df_merged[['distance']])
    else:
        df_merged['distance_score'] = 1.0 # 값이 하나뿐이면 최고 점수 부여

    # 날씨 점수도 값이 여러 개일 때만 정규화 수행
    if len(df_merged['WeatherScore'].unique()) > 1:
        df_merged['weather_score_normalized'] = scaler2.fit_transform(df_merged[['WeatherScore']])
    else:
        df_merged['weather_score_normalized'] = 1.0 # 값이 하나뿐이면 최고 점수 부여
        
    # 가중치 설정
    distance_weight = 0.7
    weather_weight = 0.3

    INDOOR_TYPE_VALUE = 1  # 실내 장소 1

    df_merged['FinalScore'] = np.where(
        df_merged['In/Out_Type(1/0)'] == INDOOR_TYPE_VALUE,  # 조건: 실내 장소일 경우
        df_merged['distance_score'],  # True: 최종 점수는 거리 점수와 동일
        (df_merged['distance_score'] * distance_weight) + (df_merged['weather_score_normalized'] * weather_weight)  
        # False: 실외는 기존 방식대로 계산
    )

    # 최종 점수가 높은 순으로 정렬하여 반환
    return df_merged.sort_values(by='FinalScore', ascending=False)

# Test
if __name__ == '__main__':
   if __name__ == '__main__':
    def format_km(val):
        if pd.isna(val):
            return "     -     "
        else:
            return f"{val:.2f}km"

    async def main():
        mbti = input('mbti를 입력하세요. : ')
        계절 = input('계절을 입력하세요. (봄/여름/가을/겨울): ')
        주소 = input('현재 위치 도로명 주소를 입력하세요: ')
        계절_map = {
            '봄': 'SEASON_SPRING', '여름': 'SEASON_SUMMER',
            '가을': 'SEASON_AUTUMN', '겨울': 'SEASON_WINTER'
        }
        계절 = 계절_map.get(계절)
        if not 계절:
            raise ValueError("계절 입력 오류")

        #추천 결과 출력
        recommendations = await recommend_places(mbti, 계절, 주소)
        print("\n--- 추천 장소 ---")
        print(recommendations[['ITS_BRO_NM', 'SIDO_NM', 'SGG_NM', 'FinalScore', 'real_distance_km']].to_string(
            index=False, justify='left', col_space=15,
            formatters={'FinalScore': '{:.6f}'.format, 'real_distance_km': format_km}))
        
        print("\n추천된 장소 중 하나의 이름을 입력하세요.")
        for name in recommendations['ITS_BRO_NM']:
            print(f"- {name}")
        selected_place = input("장소명을 정확히 입력하세요: ")

        start_coords = get_coordinates_unified(주소, is_address=True)
        end_coords = get_coordinates_unified(selected_place, is_address=False)


        if not start_coords or not end_coords:
            print("❌ 좌표 변환 실패")
            return

        route_coords = get_driving_distance(start_coords, end_coords)
        if not route_coords:
            print("❌ 경로 계산 실패 - 경로가 반환되지 않았습니다.")
            return
        
        route_coords = get_route_coordinates(start_coords, end_coords)
        if not route_coords:
            print("❌ 경로 계산 실패 - 경로가 반환되지 않았습니다.")
            return

        midpoints = [route_coords[len(route_coords)//2]]
        cafes = []
        for lon, lat in midpoints:
         cafes += search_places('CE7', lon, lat)

        m = create_map_kakao(start_coords, end_coords, route_coords, cafes)  # ✅ 수정된 부분
        m.save("selected_route_map.html")
        webbrowser.open("selected_route_map.html")

    asyncio.run(main())


'''
      # 중간지점 기준 카페 검색

    날씨는 api를 통해 불러온 후 데이터에 있는 날씨와 비교(즉, 입력받지 않음)
    하늘상태(SKY) 코드 : 맑음(1), 구름많음(3), 흐림(4)
    강수형태(PTY) 코드 : (단기) 없음(0), 비(1), 비/눈(2), 눈(3), 소나기(4)

    mbti의 경우 근거 자료로 높음: 1, 중간: 0.5, 낮음 0으로 가중치 설정

    날씨는 선정된 장소를 바탕으로 계산하고 추천해줌.
    장소 후보군을 많이 선정한 후 
    1. 근시일 내의 날씨와 관계없이 사용자 특성에 맞는 갈만한 곳 추천
    2. 날씨 특성을 추가하여 사용자 특성 & 날씨를 결합한 갈만한 곳 추천 
'''


'''
features_to_analyze = place_features.iloc[:, 1:-1] # 마지막 'cluster' 열도 제외
# 상관관계 행렬 계산
correlation_matrix = features_to_analyze.corr()
plt.rc('font', family='Malgun Gothic') # 예시: Windows
plt.rc('axes', unicode_minus=False) # 마이너스 부호 깨짐 방지
plt.figure(figsize=(18, 15)) # 히트맵 크기 조절
sns.heatmap(correlation_matrix,
            annot=True,      # 각 셀에 값 표시 여부 (특성이 많으면 False나 fmt 사용)
            cmap='coolwarm', # 색상 맵 (양/음수 표현에 적합)
            fmt='.1f',       # 소수점 자리수 (annot=True일 때)
            linewidths=.5,   # 셀 사이 경계선
            linecolor='black')# 경계선 색상
plt.title('Feature Correlation Heatmap', fontsize=20)

plt.show()
'''
