import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import request_weather
import asyncio

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
async def recommend_places(mbti: str, 계절: str):
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
    user_umap_point = user_reduced[0]  # 사용자의 UMAP 공간 좌표

    N_CLOSEST_CLUSTERS = 3 # 가장 가까운 클러스터 3개 선택
    K_ITEMS_PER_CLUSTER = 4 # 각 클러스터에서 상위 4개 장소 선택
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

        top_k_in_cluster = df_current_cluster_season.sort_values('distance').head(K_ITEMS_PER_CLUSTER)
        recommended_places_list.append(top_k_in_cluster)

    final_recommendations_df = pd.concat(recommended_places_list).sort_values('distance')

    # 날씨 정보를 받아오고 이를 기존 df에 병합
    weather_results = await request_weather.get_weather_for_dataframe_async(final_recommendations_df)
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
        df_merged['In/Out_Type(1/0)'] == INDOOR_TYPE_VALUE,  # 
        df_merged['distance_score'],  # 조건: 실내일 경우 최종 점수는 거리 점수와 동일, 실외는 날씨 점수 반영
        (df_merged['distance_score'] * distance_weight) + (df_merged['weather_score_normalized'] * weather_weight)  
    )
    return df_merged.sort_values(by='FinalScore', ascending=False)

# Test
if __name__ == '__main__':
    async def main():
        mbti = input('mbti를 입력하세요. : ')
        계절 = input('계절을 입력하세요. : ')
        if(계절 == '봄'): 계절 = 'SEASON_SPRING'
        elif(계절 == '여름'): 계절 = 'SEASON_SUMMER'
        elif(계절 == '가을'): 계절 = 'SEASON_AUTUMN'
        elif(계절 == '겨울'): 계절 = 'SEASON_WINTER'
        else: raise ValueError(f"계절 유형 '{계절}'을 찾을 수 없습니다.")

        recommendations = await recommend_places(mbti, 계절)
        print("\n--- 추천 장소 ---")
        print(recommendations)

    asyncio.run(main())


'''
    날씨는 api를 통해 불러온 후 데이터에 있는 날씨와 비교(즉, 입력받지 않음)
    하늘상태(SKY) 코드 : 맑음(1), 구름많음(3), 흐림(4)
    강수형태(PTY) 코드 : (단기) 없음(0), 비(1), 비/눈(2), 눈(3), 소나기(4)

    mbti의 경우 근거 자료로 높음: 1, 중간: 0.5, 낮음 0으로 가중치 설정

    날씨는 선정된 장소를 바탕으로 계산하고 추천해줌.
    장소 후보군을 많이 선정한 후 
    1. 근시일 내의 날씨와 관계없이 사용자 특성에 맞는 갈만한 곳 추천
    2. 날씨 특성을 추가하여 사용자 특성 & 날씨를 결합한 갈만한 곳 추천 
'''