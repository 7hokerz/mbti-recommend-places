import os
import pandas as pd
import numpy as np
import asyncio

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt

import joblib

import request_weather

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# 1. 데이터 불러오기
data = pd.read_csv(f"{script_dir}/data/place3_3.csv")

processed_data = data.copy() # 복사본

# 표준화
scaler = StandardScaler() # 스케일링

# 차원 축소
reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=3, #random_state=42
    )

# 클러스터링
k = 8 # 클러스터 수 
kmeans = KMeans(
    n_clusters=k, #random_state=42
    )

# 차원 축소 (시각화용)
tsne = TSNE(n_components=2, perplexity=40, random_state=42)

# 새로운 특성 부여
def create_features(df):
    # 키워드 특성
    df['레저/스포츠'] = df['TMAP_CATE_MCLS_NM'].apply(lambda x: 1 if '레저/스포츠' in x else 0)
    df['전통/역사'] = df['TMAP_CATE_MCLS_NM'].apply(lambda x: 1 if '전통/역사' in x else 0) 
    df['감성/체험거리'] = df['TMAP_CATE_MCLS_NM'].apply(lambda x: 1 if '감성/체험 거리' in x else 0)
    df['조망/전망'] = df['TMAP_CATE_MCLS_NM'].apply(lambda x: 1 if '조망/전망' in x else 0)
    df['자연물'] = df['TMAP_CATE_MCLS_NM'].apply(lambda x: 1 if '자연물' in x else 0)
    df['문화시설감상'] = df['TMAP_CATE_MCLS_NM'].apply(lambda x: 1 if '문화시설감상' in x else 0)

    # 키워드 간 상호작용 특성
    df['자연_조망_결합'] = (df['자연물'] & df['조망/전망']).astype(int)
    df['자연_역사_결합'] = (df['자연물'] & df['전통/역사']).astype(int)
    df['자연_레저_결합'] = (df['자연물'] & df['레저/스포츠']).astype(int)
    df['자연_체험_결합'] = (df['자연물'] & df['감성/체험거리']).astype(int)

    df['조망_체험_결합'] = (df['조망/전망'] & df['감성/체험거리']).astype(int)
    df['조망_역사_결합'] = (df['조망/전망'] & df['전통/역사']).astype(int)
    
    df['역사_문화_결합'] = (df['전통/역사'] & df['문화시설감상']).astype(int)
    df['레저_체험_결합'] = (df['레저/스포츠'] & df['감성/체험거리']).astype(int)
    
    return df

place_features = create_features(processed_data)

place_features = processed_data[[
    'SIDO_NM', 'SGG_NM', 'ITS_BRO_NM', 'In/Out_Type(1/0)',
    'SEASON_SPRING','SEASON_SUMMER','SEASON_AUTUMN','SEASON_WINTER',
    '레저/스포츠', '전통/역사', '감성/체험거리', '조망/전망', '자연물', '문화시설감상',
    '자연_조망_결합','자연_역사_결합','자연_레저_결합','자연_체험_결합','역사_문화_결합','레저_체험_결합',
    '조망_체험_결합','조망_역사_결합'
]].copy()

# 2. 데이터 전처리
# 2.1 표준화
X = scaler.fit_transform(place_features.iloc[:, 8:])

# 2.2 차원 축소
X_reduced = reducer.fit_transform(X, ensure_all_finite=True)

# 2.3 클러스터링
clusters = kmeans.fit_predict(X_reduced) 
place_features['cluster'] = clusters

# 3. 클러스터링 결과 평가
print(f"Silhouette score:{silhouette_score(X_reduced, place_features['cluster'])}")
print(f"SSE (Inertia): {kmeans.inertia_}")
db_score = davies_bouldin_score(X_reduced, kmeans.labels_)
print(f"Davies-Bouldin Index: {db_score}")
ch_score = calinski_harabasz_score(X_reduced, kmeans.labels_)
print(f"Calinski-Harabasz Index: {ch_score}")

#for i in range(8):
#        print(f'cluster{i}: {place_features[place_features['cluster'] == i]['cluster'].count()}')

# 4. 클러스터 시각화 (t-SNE)
def visualize():
    X_tsne = tsne.fit_transform(X_reduced)
    place_features['x'] = X_tsne[:, 0] 
    place_features['y'] = X_tsne[:, 1]

    plt.figure(figsize=(10, 8))
    for i in range(k):
        cluster_points = place_features[place_features['cluster'] == i]
        plt.scatter(cluster_points['x'], cluster_points['y'], label=f'Cluster {i}')
    plt.title('place clustering')
    plt.legend()
    plt.show()

# visualize()

mbti_weights = {
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

# 장소 추천 함수
async def recommend_places(mbti, 계절):
    get_weight = mbti_weights[mbti]
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
        자연_조망_결합, 자연_역사_결합, 자연_레저_결합, 자연_체험_결합, 역사_문화_결합, 레저_체험_결합,
        조망_체험_결합, 조망_역사_결합
    ]]
    user_features_df = pd.DataFrame(user_features_data, columns=scaler.feature_names_in_)

    # 스케일링 > UMAP > 모델 예측
    user_scaled = scaler.transform(user_features_df)
    user_reduced = reducer.transform(user_scaled)
    user_umap_point = user_reduced[0]  # 사용자의 UMAP 공간 좌표

    cluster_centroids = kmeans.cluster_centers_

    distances_to_centroids = np.sqrt(np.sum((cluster_centroids - user_umap_point)**2, axis=1))

    N_CLOSEST_CLUSTERS = 3  # 예: 가장 가까운 클러스터 3개 선택
    K_ITEMS_PER_CLUSTER = 5 # 예: 각 클러스터에서 상위 5개 장소 선택

    closest_cluster_labels = np.argsort(distances_to_centroids)[:N_CLOSEST_CLUSTERS]

    # 거리 계산 및 장소 선택
    recommended_places_list = []
    
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

    weather_results = await request_weather.get_weather_for_dataframe_async(final_recommendations_df)

    df_current = pd.DataFrame(pd.Series(weather_results), columns=['data_dict'])

    df_expanded = df_current['data_dict'].apply(pd.Series)

    df_expanded.index = final_recommendations_df.index

    df_merged = final_recommendations_df.join(df_expanded)

    df_merged = df_merged.sort_values('맑음', ascending=False)
    
    return df_merged

# recommend_places('ISTJ', 'SEASON_SPRING')


'''

[[
#  'SIDO_NM', 'SGG_NM','ITS_BRO_NM','In/Out_Type(1/0)','맑음','구름많음','흐림','없음','비','소나기'
#   ]])
    날씨는 api를 통해 불러온 후 데이터에 있는 날씨와 비교(즉, 입력받지 않음)
    하늘상태(SKY) 코드 : 맑음(1), 구름많음(3), 흐림(4)
    강수형태(PTY) 코드 : (단기) 없음(0), 비(1), 비/눈(2), 눈(3), 소나기(4)

    mbti의 경우 근거 자료로 높음: 1, 중간: 0.75, 낮음 0.5로 가중치 설정

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