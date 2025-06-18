from itertools import product
import pandas as pd
import numpy as np
import joblib
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage # << 덴드로그램용 추가

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 디렉토리 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

# 기본 설정값
N_CLUSTERS = 8 # 클러스터 수 
UMAP_PARAMS = {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 3 }
KMEANS_PARAMS = {'n_clusters': N_CLUSTERS }
TSNE_PARAMS = {'n_components': 2, 'perplexity': 40 }
AGG_PARAMS = {'n_clusters': 10, 'linkage': 'ward'} # << 계층적 군집화 추가

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

    scaler = StandardScaler() # 표준화 스케일링
    reducer = umap.UMAP(**UMAP_PARAMS) # 차원 축소
    kmeans = KMeans(**KMEANS_PARAMS) # 클러스터링
    tsne = TSNE(**TSNE_PARAMS) # 차원 축소 (시각화용)
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    agg_cluster = AgglomerativeClustering(**AGG_PARAMS) # << 계층적 군집화 추가
    
    X = scaler.fit_transform(place_features.iloc[:, 8:]) # 표준화 스케일링
    X_reduced = reducer.fit_transform(X, ensure_all_finite=True) # 차원 축소
    clusters = kmeans.fit_predict(X_reduced) # 클러스터링
    dbscan.fit(X_reduced)
    agg_clusters = agg_cluster.fit_predict(X_reduced) # << 계층적 군집화 추가
    place_features['cluster'] = clusters
    place_features['cluster_agg'] = agg_clusters # << 계층적 군집화 추가

    print("3. 모델 및 처리된 데이터 저장 중...")
    # 모델 저장
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(reducer, os.path.join(MODEL_DIR, 'reducer.pkl'))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans.pkl'))
    joblib.dump(tsne, os.path.join(MODEL_DIR, 'tsne.pkl'))
    joblib.dump(dbscan, os.path.join(MODEL_DIR, 'dbscan.pkl'))
    joblib.dump(agg_cluster, os.path.join(MODEL_DIR, 'agg_cluster.pkl'))

     # 처리된 데이터와 차원 축소된 numpy 배열 저장
    np.save(os.path.join(MODEL_DIR, 'X_reduced.npy'), X_reduced)
    place_features.to_csv(os.path.join(MODEL_DIR, 'clustered_places.csv'), index=False)
    print("학습 완료. 모델과 데이터가 'models' 디렉토리에 저장되었습니다.")

# 평가 지표 출력
def evaluation_metrics(): 
    try:
        kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
        dbscan = joblib.load(os.path.join(MODEL_DIR, 'dbscan.pkl'))
        agg_cluster = joblib.load(os.path.join(MODEL_DIR, 'agg_cluster.pkl'))
        X_reduced = np.load(os.path.join(MODEL_DIR, 'X_reduced.npy'))
        place_features = pd.read_csv(os.path.join(MODEL_DIR, 'clustered_places.csv'))
    except FileNotFoundError:
        print("오류: 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
        exit()


    # --- KMeans 평가 지표 ---
    print("--- K-Means 평가 지표 ---")

    si_score = silhouette_score(X_reduced, place_features['cluster'])
    db_score = davies_bouldin_score(X_reduced, kmeans.labels_)
    ch_score = calinski_harabasz_score(X_reduced, kmeans.labels_)
    
    print(f"Silhouette score:{si_score}")
    print(f"SSE (Inertia): {kmeans.inertia_}")
    print(f"Davies-Bouldin Index: {db_score}")
    print(f"Calinski-Harabasz Index: {ch_score}")


    # --- DBSCAN 평가 지표 ---
    print("--- DBSCAN 평가 지표 ---")

    labels = dbscan.labels_
    core_points_mask = labels != -1
    X_clustered = X_reduced[core_points_mask]
    labels_clustered = labels[core_points_mask]

    # 클러스터가 2개 이상일 때만 평가지표 계산이 가능합니다.
    if len(set(labels_clustered)) < 2:
        print("DBSCAN 결과, 유효한 클러스터가 1개 이하이므로 평가지표를 계산할 수 없습니다.")
        # 노이즈 비율이라도 출력해주는 것이 유용합니다.
        noise_ratio = list(labels).count(-1) / len(labels)
        print(f"노이즈 데이터 비율: {noise_ratio:.2%}")
    else:
        # 모든 평가지표를 노이즈가 제거된 데이터와 라벨로 계산합니다.
        si_score_d = silhouette_score(X_clustered, labels_clustered)
        db_score_d = davies_bouldin_score(X_clustered, labels_clustered)
        ch_score_d = calinski_harabasz_score(X_clustered, labels_clustered)
        
        print(f"Silhouette score (노이즈 제외): {si_score_d}")
        # DBSCAN에는 SSE(Inertia)가 없으므로 해당 라인을 삭제합니다.
        print(f"Davies-Bouldin Index (노이즈 제외): {db_score_d}")
        print(f"Calinski-Harabasz Index (노이즈 제외): {ch_score_d}")

    # --- 계층적 군집화 평가 지표 ---
    print("\n--- 계층적 군집화(Agglomerative) 평가 지표 ---") # << 계층적 군집화 추가
    labels_agg = agg_cluster.labels_
    si_score_a = silhouette_score(X_reduced, labels_agg)
    db_score_a = davies_bouldin_score(X_reduced, labels_agg)
    ch_score_a = calinski_harabasz_score(X_reduced, labels_agg)
    print(f"Silhouette score: {si_score_a}")
    print(f"Davies-Bouldin Index: {db_score_a}")
    print(f"Calinski-Harabasz Index: {ch_score_a}")

# --- 덴드로그램 시각화 (새로운 함수) ---
def visualize_dendrogram(): # << 계층적 군집화 추가
    """저장된 데이터를 이용해 덴드로그램을 시각화합니다."""
    print("\n--- 덴드로그램 생성 중 ---")
    try:
        X_reduced = np.load(os.path.join(MODEL_DIR, 'X_reduced.npy'))
    except FileNotFoundError:
        print("오류: X_reduced.npy 파일을 찾을 수 없습니다. 'train_save_models()'를 먼저 실행해주세요.")
        return

    # 'ward' 연결법을 사용하여 linkage 행렬 생성
    linked = linkage(X_reduced, method='ward')

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15, 7))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False) # leaf 수가 많으면 복잡하므로 False
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance (Ward)')
    plt.suptitle("덴드로그램을 보고 y축(거리)을 기준으로 선을 그어 클러스터 수를 결정할 수 있습니다.", y=0.92)
    plt.show()

# 엘보우 테스트
def elbow_search(max_k=10):
    try:
        X_reduced = np.load(os.path.join(MODEL_DIR, 'X_reduced.npy'))
    except FileNotFoundError:
        print("오류: 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
        exit()
    
    # 여러 K에 대한 성능 지표
    k_range = range(2, max_k + 1)
    
    inertia_list = []
    silhouette_list = []
    davies_bouldin_list = []
    calinski_harabasz_list = []
    
    # k값을 2부터 max_k까지 성능 평가
    print(f"k=2부터 {max_k}까지의 클러스터 성능을 평가합니다...")
    for k in k_range:
        # K-Means 모델 학습
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_reduced)
        labels = kmeans.labels_

        # 평가지표 계산 및 저장
        inertia_list.append(kmeans.inertia_)
        silhouette_list.append(silhouette_score(X_reduced, labels))
        davies_bouldin_list.append(davies_bouldin_score(X_reduced, labels))
        calinski_harabasz_list.append(calinski_harabasz_score(X_reduced, labels))
    
    # 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    # 4개의 평가지표를 한 번에 시각화
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('다중 평가지표를 이용한 최적의 k 찾기', fontsize=16)

    # 1. 이너셔 (Elbow Method)
    axs[0, 0].plot(k_range, inertia_list, 'o-')
    axs[0, 0].set_title('이너셔 (SSE)')
    axs[0, 0].set_xlabel('클러스터 개수 (k)')
    axs[0, 0].set_ylabel('Inertia')

    # 2. 실루엣 점수
    axs[0, 1].plot(k_range, silhouette_list, 'o-')
    axs[0, 1].set_title('실루엣 점수 ( 높을수록 좋음)')
    axs[0, 1].set_xlabel('클러스터 개수 (k)')
    axs[0, 1].set_ylabel('Silhouette Score')

    # 3. 데이비스-볼딘 점수
    axs[1, 0].plot(k_range, davies_bouldin_list, 'o-')
    axs[1, 0].set_title('데이비스-볼딘 점수 (작을수록 좋음)')
    axs[1, 0].set_xlabel('클러스터 개수 (k)')
    axs[1, 0].set_ylabel('Davies-Bouldin Score')

    # 4. 칼린스키-하라바츠 점수
    axs[1, 1].plot(k_range, calinski_harabasz_list, 'o-')
    axs[1, 1].set_title('칼린스키-하라바츠 점수 (높을수록 좋음)')
    axs[1, 1].set_xlabel('클러스터 개수 (k)')
    axs[1, 1].set_ylabel('Calinski-Harabasz Score')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 그리드 서치
def elbow_search2(fixed_k=8):
    data = pd.read_csv(f"{SCRIPT_DIR}/data/place3_3.csv") # 데이터 로드
    place_features = create_features(data.copy())[[
        'SIDO_NM', 'SGG_NM', 'ITS_BRO_NM', 'In/Out_Type(1/0)',
        'SEASON_SPRING','SEASON_SUMMER','SEASON_AUTUMN','SEASON_WINTER',
        '레저/스포츠', '전통/역사', '감성/체험거리', '조망/전망', '자연물', '문화시설감상',
        '자연_조망_결합','자연_역사_결합','자연_레저_결합','자연_체험_결합',
        '역사_문화_결합','레저_체험_결합','조망_체험_결합','조망_역사_결합'
    ]].copy() # 특성 공학
    
    # 표준화
    scaler = StandardScaler() # 표준화 스케일링
    X = scaler.fit_transform(place_features.iloc[:, 8:]) # 표준화 스케일링
    
    #리스트 초기화
    inertia_list = []
    silhouette_list = []
    davies_bouldin_list = []
    calinski_harabasz_list = []
    
    # UMAP 파라미터 조합
    param_grid = {
        'n_neighbors': [15, 30],
        'min_dist': [0.1]
    }
    
    # 조합 리스트
    param_combinations = list(product(param_grid['n_neighbors'], param_grid['min_dist']))
    
    # 파라미터 성능 평가
    for n_neighbors, min_dist in param_combinations:
        # 1. UMAP으로 데이터 변환 (임베딩)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=3, # 예시로 3차원으로 설정
        )
        X_reduced = reducer.fit_transform(X)

        # 2. 고정된 k로 K-Means 모델 학습
        kmeans = KMeans(n_clusters=fixed_k)
        kmeans.fit(X_reduced)
        labels = kmeans.labels_

        # 3. 평가지표 계산 및 저장
        # 모든 점수는 UMAP으로 변환된 'X_reduced'를 기준으로 계산해야 함
        inertia_list.append(kmeans.inertia_)
        if len(np.unique(labels)) > 1: # 클러스터가 2개 이상일 때만 점수 계산
            silhouette_list.append(silhouette_score(X_reduced, labels))
            davies_bouldin_list.append(davies_bouldin_score(X_reduced, labels))
            calinski_harabasz_list.append(calinski_harabasz_score(X_reduced, labels))
        else: # 클러스터링 실패 시
            silhouette_list.append(-1)
            davies_bouldin_list.append(np.inf)
            calinski_harabasz_list.append(0)
    
    # 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 서브플롯 생성
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'UMAP 파라미터 조합에 따른 클러스터링 성능 평가 (k={fixed_k})', fontsize=16)
    
    # X축 레이블로 사용할 파라미터 조합 문자열 생성
    param_labels = [f"n:{n}\nd:{d}" for n, d in param_combinations]

    # 1. 이너셔 (Elbow Method) - 낮을수록 좋음
    axs[0, 0].plot(range(len(param_labels)), inertia_list, 'o-')
    axs[0, 0].set_title('이너셔 (SSE) - 낮을수록 좋음')
    axs[0, 0].set_ylabel('Inertia')

    # 2. 실루엣 점수 - 높을수록 좋음
    axs[0, 1].plot(range(len(param_labels)), silhouette_list, 'o-')
    axs[0, 1].set_title('실루엣 점수 - 높을수록 좋음')
    axs[0, 1].set_ylabel('Silhouette Score')

    # 3. 데이비스-볼딘 점수 - 작을수록 좋음
    axs[1, 0].plot(range(len(param_labels)), davies_bouldin_list, 'o-')
    axs[1, 0].set_title('데이비스-볼딘 점수 - 작을수록 좋음')
    axs[1, 0].set_ylabel('Davies-Bouldin Score')

    # 4. 칼린스키-하라바츠 점수 - 높을수록 좋음
    axs[1, 1].plot(range(len(param_labels)), calinski_harabasz_list, 'o-')
    axs[1, 1].set_title('칼린스키-하라바츠 점수 - 높을수록 좋음')
    axs[1, 1].set_ylabel('Calinski-Harabasz Score')

    # 모든 subplot에 x축 레이블 설정
    for ax in axs.flat:
        ax.set_xlabel('UMAP 파라미터 조합 (n_neighbors, min_dist)')
        ax.set_xticks(range(len(param_labels)))
        ax.set_xticklabels(param_labels, rotation=45, ha="right")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 클러스터 시각화 (t-SNE)
def visualize():
    try:
        # 학습된 모델, 클러스터 결과 불러오기
        tsne = joblib.load(os.path.join(MODEL_DIR, 'tsne.pkl'))
        X_reduced = np.load(os.path.join(MODEL_DIR, 'X_reduced.npy'))
        place_features = pd.read_csv(os.path.join(MODEL_DIR, 'clustered_places.csv'))
    except FileNotFoundError:
        print("오류: 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
        exit()
    # 축소된 데이터를 t-SNE로 추가 축소    
    X_tsne = tsne.fit_transform(X_reduced)
    place_features['x'] = X_tsne[:, 0] 
    place_features['y'] = X_tsne[:, 1]
    
    # 산점도 그리기
    plt.figure(figsize=(10, 8))
    for i in range(N_CLUSTERS):
        cluster_points = place_features[place_features['cluster'] == i]
        plt.scatter(cluster_points['x'], cluster_points['y'], label=f'Cluster {i}')
    plt.title('place clustering')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_save_models() # 모델 학습 저장
    #elbow_search()
    #elbow_search2()
    evaluation_metrics() # 평가 지표 출력
    #visualize_dendrogram()
    #visualize()
