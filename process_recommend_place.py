import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import request_weather
import asyncio
import webbrowser
import tabulate
from kakao_route_service import get_coordinates_unified, get_driving_distance, create_map_kakao, search_places, get_route_coordinates

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

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

WEATHER_WEIGHTS = {
    '맑음': 1.0,
    '구름많음': 1.0,
    '흐림': 1.0,
    '비': -1.0,
    '비/눈': -1.0,
    '눈': -1.0,
    '소나기': -1.0
}

# 저장된 모델과 클러스터링 분류 데이터 가져오기
try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    reducer = joblib.load(os.path.join(MODEL_DIR, 'reducer.pkl'))
    kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
    dbscan = joblib.load(os.path.join(MODEL_DIR, 'dbscan.pkl'))
    agg_cluster = joblib.load(os.path.join(MODEL_DIR, 'agg_cluster.pkl'))
    X_reduced = np.load(os.path.join(MODEL_DIR, 'X_reduced.npy'))
    place_features = pd.read_csv(os.path.join(MODEL_DIR, 'clustered_places.csv'))
except FileNotFoundError:
    print("오류: 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
    exit()

# 장소 추천 함수
async def recommend_places(mbti: str, 계절: str, 현재주소: str, 날짜: str):
    get_weight = MBTI_WEIGHTS.get(mbti.upper()) # 대문자 변환
    if not get_weight:
        raise ValueError(f"MBTI 유형 '{mbti}'를 찾을 수 없습니다.")

    # 사용자의 특성 점수 계산
    레저_스포츠 = get_weight["레저/스포츠"]
    전통_역사 = get_weight["전통/역사"]
    감성_체험거리 = get_weight["감성/체험거리"]
    조망_전망 = get_weight["조망/전망"]
    자연물 = get_weight["자연물"]
    문화시설감상 = get_weight["문화시설감상"]
    자연_조망_결합 = 자연물 * 조망_전망
    자연_역사_결합 = 자연물 * 전통_역사
    자연_레저_결합 = 자연물 * 레저_스포츠
    자연_체험_결합 = 자연물 * 감성_체험거리
    조망_체험_결합 = 조망_전망 * 감성_체험거리
    조망_역사_결합 = 조망_전망 * 전통_역사
    역사_문화_결합 = 전통_역사 * 문화시설감상
    레저_체험_결합 = 레저_스포츠 * 감성_체험거리

    user_features_data = [[
        레저_스포츠, 전통_역사, 감성_체험거리, 조망_전망, 자연물, 문화시설감상,
        자연_조망_결합, 자연_역사_결합, 자연_레저_결합, 자연_체험_결합, 
        역사_문화_결합, 레저_체험_결합, 조망_체험_결합, 조망_역사_결합
    ]]
    user_features_df = pd.DataFrame(user_features_data, columns=scaler.feature_names_in_)

    # 스케일링 > UMAP > 모델 예측
    user_scaled = scaler.transform(user_features_df)
    user_reduced = reducer.transform(user_scaled)
    user_umap_point = user_reduced[0] # 사용자의 UMAP 공간 좌표

    N_CLOSEST_CLUSTERS = 4 # 가장 가까운 클러스터 3개 선택
    K_ITEMS_PER_CLUSTER = 3 # 각 클러스터에서 상위 4개 장소 선택
    # 중심점으로부터 거리 계산 및 가까운 클러스터 선정
    cluster_centroids = kmeans.cluster_centers_
    distances_to_centroids = np.sqrt(np.sum((cluster_centroids - user_umap_point)**2, axis=1))
    closest_cluster_labels = np.argsort(distances_to_centroids)[:N_CLOSEST_CLUSTERS]
    
    recommended_places_list = []
    # 각 클러스터별 장소 간 거리 계산 및 정렬
    for cluster_label in closest_cluster_labels:
        df_current_cluster_season = place_features[
            (place_features['cluster'] == cluster_label) &
            (place_features[계절] == 1)].copy()
        
        if df_current_cluster_season.empty:
            continue
        
        indices_in_X_reduced = df_current_cluster_season.index
        points_for_distance_calc = X_reduced[indices_in_X_reduced]
        
        distances_per_place = np.sqrt(np.sum((points_for_distance_calc - user_umap_point)**2, axis=1))
        df_current_cluster_season['distance'] = distances_per_place
        # 최종 K_ITEMS_PER_CLUSTER 만큼 반환
        top_k_in_cluster = df_current_cluster_season.sort_values('distance').head(K_ITEMS_PER_CLUSTER)
        recommended_places_list.append(top_k_in_cluster)

    final_recommendations_df = pd.concat(recommended_places_list).sort_values('distance')

    # 실제 거리 계산 (수정한 부분)
    final_recommendations_df['real_distance_km'] = final_recommendations_df['ITS_BRO_NM'].apply(
    lambda place: get_driving_distance(현재주소, place)
    )

    # 날씨 정보를 받아오고 이를 기존 df에 병합
    weather_results = await request_weather.get_weather_for_dataframe_async(final_recommendations_df, 날짜)

    # 통신 문제 등으로 인해 날씨 정보를 받지 못하면 거리 기준으로만 반환
    if weather_results is not None:
        df_current = pd.DataFrame(pd.Series(weather_results), columns=['data_dict'])
        df_expanded = df_current['data_dict'].apply(pd.Series)
        df_expanded.index = final_recommendations_df.index
        df_merged = final_recommendations_df.join(df_expanded)

        # 날씨 점수 계산 및 정규화
        scaler2 = MinMaxScaler()
        weather_cols = [col for col in WEATHER_WEIGHTS.keys() if col in df_merged.columns]
        df_merged['WeatherScore'] = sum(df_merged[col] * WEATHER_WEIGHTS[col] for col in weather_cols)

        # 거리 값이 여러 개일 때만 정규화 수행
        if len(df_merged['distance'].unique()) > 1:
            df_merged['distance_score'] = 1 - scaler2.fit_transform(df_merged[['distance']])
        else:
            df_merged['distance_score'] = 1.0

        # 날씨 점수 값이 여러 개일 때만 정규화 수행
        if len(df_merged['WeatherScore'].unique()) > 1:
            df_merged['weather_score_normalized'] = scaler2.fit_transform(df_merged[['WeatherScore']])
        else:
            df_merged['weather_score_normalized'] = 1.0
        
        distance_weight = 0.7
        weather_weight = 0.3
        INDOOR_TYPE_VALUE = 1
        # 날씨 점수는 실외 장소에만 반영하고 최종 점수가 높은 순으로 정렬하여 반환
        df_merged['FinalScore'] = np.where(
            df_merged['In/Out_Type(1/0)'] == INDOOR_TYPE_VALUE,
            df_merged['distance_score'],  # 조건: 실내일 경우 최종 점수는 거리 점수와 동일, 실외는 날씨 점수 반영
            (df_merged['distance_score'] * distance_weight) + (df_merged['weather_score_normalized'] * weather_weight)  
        )
        return df_merged.sort_values(by='FinalScore', ascending=False)
    else:
        return final_recommendations_df

def show_route(recommendations, 주소):
    print("\n--- 추천 장소 목록 ---")

    print(tabulate.tabulate(
    recommendations.dropna(subset=['real_distance_km'])[['ITS_BRO_NM', 'SIDO_NM', 'SGG_NM','WeatherScore','FinalScore', 'real_distance_km']],
    headers='keys', tablefmt='pretty'))
    print("\n추천된 장소 리스트 입니다. (경로 추천 가능)\n")
    for idx, row in recommendations.iterrows():
        if pd.notna(row['real_distance_km']):
            print(f"- {row['ITS_BRO_NM']}")

    sys.stdout.flush()
    
    selected_place = input("도착지로 설정할 장소명을 정확히 입력하세요 : ")

    start_coords = get_coordinates_unified(주소, is_address=True)
    end_coords = get_coordinates_unified(selected_place, is_address=False)

    route_coords = get_route_coordinates(start_coords, end_coords)
    if not route_coords:
        print("경로 계산 실패 - 경로가 반환되지 않았습니다.")
        return
    
    search_categories = {
        'CE7': '카페',
        'FD6': '음식점',
        'AT4': '관광명소'
    }

    end_lon, end_lat = end_coords
    all_nearby_places = [] # 모든 검색 결과를 담을 리스트

    print(f"\n--- '{selected_place}' 주변 장소 검색 ---")
    # 2. 정의된 카테고리 목록을 순회하며 장소를 검색하고 리스트에 추가합니다.
    for code, description in search_categories.items():
        print(f"-> 주변 {description} 목록을 검색합니다...")
        # search_places 함수를 호출하여 결과를 받아옴
        places = search_places(code, end_lon, end_lat)
        all_nearby_places.extend(places) # 검색 결과를 전체 리스트에 추가

    # 3. 통합된 장소 목록을 지도 생성 함수에 전달합니다.
    m = create_map_kakao(start_coords, end_coords, route_coords, all_nearby_places)

    m.save("selected_route_map.html")
    webbrowser.open("selected_route_map.html")

# Test
if __name__ == '__main__':
    async def main():
        mbti = input('mbti를 입력하세요 : ')
        계절 = input('계절을 입력하세요(봄/여름/가을/겨울) : ')
        주소 = input('출발지의 주소를 입력하세요 (예: 서울특별시 성북구 삼선교로16길 116): ')
        날짜 = input('가고 싶은 날짜 (현재로부터 4일 이내)를 입력하세요 (예: 20250619): ')
        계절_map = {
            '봄': 'SEASON_SPRING', '여름': 'SEASON_SUMMER',
            '가을': 'SEASON_AUTUMN', '겨울': 'SEASON_WINTER'
        }
        계절 = 계절_map.get(계절)
        if not 계절:
            raise ValueError("계절 입력 오류") 

        recommendations = await recommend_places(mbti, 계절, 주소, 날짜)
        show_route(recommendations, 주소)

    asyncio.run(main())