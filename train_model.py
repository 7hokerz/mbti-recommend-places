import pandas as pd
import numpy as np
import joblib
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

N_CLUSTERS = 8 # 클러스터 수 
UMAP_PARAMS = {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 3 }
KMEANS_PARAMS = {'n_clusters': N_CLUSTERS }
TSNE_PARAMS = {'n_components': 2, 'perplexity': 40 }

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

# 모델 학습 및 저장
def train_save_models():
    data = pd.read_csv(f"{SCRIPT_DIR}/data/place3_3.csv")
    place_features = create_features(data.copy())[[
        'SIDO_NM', 'SGG_NM', 'ITS_BRO_NM', 'In/Out_Type(1/0)',
        'SEASON_SPRING','SEASON_SUMMER','SEASON_AUTUMN','SEASON_WINTER',
        '레저/스포츠', '전통/역사', '감성/체험거리', '조망/전망', '자연물', '문화시설감상',
        '자연_조망_결합','자연_역사_결합','자연_레저_결합','자연_체험_결합',
        '역사_문화_결합','레저_체험_결합','조망_체험_결합','조망_역사_결합'
    ]].copy() # 특성 공학

    scaler = StandardScaler() # 표준화 스케일링?
    reducer = umap.UMAP(**UMAP_PARAMS) # 차원 축소
    kmeans = KMeans(**KMEANS_PARAMS) # 클러스터링
    tsne = TSNE(**TSNE_PARAMS) # 차원 축소 (시각화용)
    
    X = scaler.fit_transform(place_features.iloc[:, 8:]) # 표준화 스케일링
    X_reduced = reducer.fit_transform(X, ensure_all_finite=True) # 차원 축소
    clusters = kmeans.fit_predict(X_reduced) # 클러스터링
    place_features['cluster'] = clusters

    print("3. 모델 및 처리된 데이터 저장 중...")
    # 모델 저장
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(reducer, os.path.join(MODEL_DIR, 'reducer.pkl'))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans.pkl'))
    joblib.dump(tsne, os.path.join(MODEL_DIR, 'tsne.pkl'))

     # 처리된 데이터와 차원 축소된 numpy 배열 저장
    np.save(os.path.join(MODEL_DIR, 'X_reduced.npy'), X_reduced)
    place_features.to_csv(os.path.join(MODEL_DIR, 'clustered_places.csv'), index=False)

    print("학습 완료. 모델과 데이터가 'models' 디렉토리에 저장되었습니다.")

# 평가 지표 출력
def evaluation_metrics(): 
    try:
        kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
        X_reduced = np.load(os.path.join(MODEL_DIR, 'X_reduced.npy'))
        place_features = pd.read_csv(os.path.join(MODEL_DIR, 'clustered_places.csv'))
    except FileNotFoundError:
        print("오류: 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
        exit()

    si_score = silhouette_score(X_reduced, place_features['cluster'])
    db_score = davies_bouldin_score(X_reduced, kmeans.labels_)
    ch_score = calinski_harabasz_score(X_reduced, kmeans.labels_)
    
    print(f"Silhouette score:{si_score}")
    print(f"SSE (Inertia): {kmeans.inertia_}")
    print(f"Davies-Bouldin Index: {db_score}")
    print(f"Calinski-Harabasz Index: {ch_score}")




# 클러스터 시각화 (t-SNE)
def visualize():
    try:
        tsne = joblib.load(os.path.join(MODEL_DIR, 'tsne.pkl'))
        X_reduced = np.load(os.path.join(MODEL_DIR, 'X_reduced.npy'))
        place_features = pd.read_csv(os.path.join(MODEL_DIR, 'clustered_places.csv'))
    except FileNotFoundError:
        print("오류: 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
        exit()
    X_tsne = tsne.fit_transform(X_reduced)
    place_features['x'] = X_tsne[:, 0] 
    place_features['y'] = X_tsne[:, 1]

    plt.figure(figsize=(10, 8))
    for i in range(N_CLUSTERS):
        cluster_points = place_features[place_features['cluster'] == i]
        plt.scatter(cluster_points['x'], cluster_points['y'], label=f'Cluster {i}')
    plt.title('place clustering')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_save_models()
    #evaluation_metrics()
    #visualize()
