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
import seaborn as sns

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

N_CLUSTERS = 12 # 클러스터 수 
UMAP_PARAMS = {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 3 } # UMAP 차원 축소
KMEANS_PARAMS = {'n_clusters': N_CLUSTERS} # KMeans 클러스터링
AGG_PARAMS = {'n_clusters': 12, 'linkage': 'ward'} # 계층적 군집화
DBSCAN_PARAMS = {'eps': 0.5, 'min_samples': 3 }
TSNE_PARAMS = {'n_components': 2, 'perplexity': 40 } # T-SNE 시각화용

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
    dbscan = DBSCAN(**DBSCAN_PARAMS)
    agg_cluster = AgglomerativeClustering(**AGG_PARAMS) # 계층적 군집화
    
    X = scaler.fit_transform(place_features.iloc[:, 8:]) # 표준화 스케일링
    X_reduced = reducer.fit_transform(X, ensure_all_finite=True) # 차원 축소
    clusters = kmeans.fit_predict(X_reduced) # 클러스터링
    agg_clusters = agg_cluster.fit_predict(X_reduced) # 계층적 군집화
    dbscan.fit(X_reduced)
    place_features['cluster'] = clusters
    place_features['cluster_agg'] = agg_clusters # 계층적 군집화

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
    except FileNotFoundError:
        print("오류: 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
        return None

    # 결과를 저장할 딕셔너리
    metrics = {}

    # --- K-Means 평가 ---
    metrics['K-Means'] = {
        'Silhouette': silhouette_score(X_reduced, kmeans.labels_),
        'Davies-Bouldin': davies_bouldin_score(X_reduced, kmeans.labels_),
        'Calinski-Harabasz': calinski_harabasz_score(X_reduced, kmeans.labels_)
    }

    # --- DBSCAN 평가 ---
    labels_db = dbscan.labels_
    core_points_mask = labels_db != -1
    labels_db_clustered = labels_db[core_points_mask]

    if len(set(labels_db_clustered)) < 2:
        print("DBSCAN: 클러스터가 1개 이하이므로 평가지표를 계산하지 않습니다.")
        metrics['DBSCAN'] = {'Silhouette': np.nan, 'Davies-Bouldin': np.nan, 'Calinski-Harabasz': np.nan}
    else:
        X_db_clustered = X_reduced[core_points_mask]
        metrics['DBSCAN'] = {
            'Silhouette': silhouette_score(X_db_clustered, labels_db_clustered),
            'Davies-Bouldin': davies_bouldin_score(X_db_clustered, labels_db_clustered),
            'Calinski-Harabasz': calinski_harabasz_score(X_db_clustered, labels_db_clustered)
        }

    # --- 계층적 군집화 평가 ---
    metrics['Hierarchical'] = {
        'Silhouette': silhouette_score(X_reduced, agg_cluster.labels_),
        'Davies-Bouldin': davies_bouldin_score(X_reduced, agg_cluster.labels_),
        'Calinski-Harabasz': calinski_harabasz_score(X_reduced, agg_cluster.labels_)
    }

    # 딕셔너리를 DataFrame으로 변환하여 반환
    metrics_df = pd.DataFrame(metrics).T
    print("--- 모델별 평가 지표 ---")
    print(metrics_df)
    return metrics_df

# 엘보우 서치 (최적의 K값 확인)
def elbow_search(max_k=13):
    try:
        X_reduced = np.load(os.path.join(MODEL_DIR, 'X_reduced.npy'))
    except FileNotFoundError:
        print("오류: 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
        exit()

    k_range = range(6, max_k + 1)
    
    inertia_list = []
    silhouette_list = []
    davies_bouldin_list = []
    calinski_harabasz_list = []

    print(f"k=2부터 {max_k}까지의 클러스터 성능을 평가합니다...")
    for k in k_range:
        # K-Means 모델 학습
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_reduced)
        labels = kmeans.labels_

        # 평가지표 계산 및 저장
        inertia_list.append(kmeans.inertia_)
        silhouette_list.append(silhouette_score(X_reduced, labels))
        davies_bouldin_list.append(davies_bouldin_score(X_reduced, labels))
        calinski_harabasz_list.append(calinski_harabasz_score(X_reduced, labels))
    
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

# 그리드 서치 (최적의 UMAP 파라미터값 확인)
def grid_search(fixed_k=8):
    data = pd.read_csv(f"{SCRIPT_DIR}/data/place3_3.csv")
    place_features = create_features(data.copy())[[
        'SIDO_NM', 'SGG_NM', 'ITS_BRO_NM', 'In/Out_Type(1/0)',
        'SEASON_SPRING','SEASON_SUMMER','SEASON_AUTUMN','SEASON_WINTER',
        '레저/스포츠', '전통/역사', '감성/체험거리', '조망/전망', '자연물', '문화시설감상',
        '자연_조망_결합','자연_역사_결합','자연_레저_결합','자연_체험_결합',
        '역사_문화_결합','레저_체험_결합','조망_체험_결합','조망_역사_결합'
    ]].copy() # 특성 공학

    scaler = StandardScaler() # 표준화 스케일링?
    X = scaler.fit_transform(place_features.iloc[:, 8:]) 
    
    inertia_list = []
    silhouette_list = []
    davies_bouldin_list = []
    calinski_harabasz_list = []

    param_grid = {
        'n_neighbors': [15, 30],
        'min_dist': [0.1]
    }

    param_combinations = list(product(param_grid['n_neighbors'], param_grid['min_dist']))

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
    
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
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

# 클러스터 시각화 2차원 (t-SNE)
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

# 클러스터 시각화 3차원 (t-SNE)
def visualize_3d():
    try:
        X_reduced = np.load(os.path.join(MODEL_DIR, 'X_reduced.npy'))
        place_features = pd.read_csv(os.path.join(MODEL_DIR, 'clustered_places.csv'))
    except FileNotFoundError:
        print("오류: 데이터 파일을 찾을 수 없습니다. 'train_model.py' 등으로 파일이 준비되었는지 확인해주세요.")
        return

    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    X_tsne_3d = tsne_3d.fit_transform(X_reduced)

    place_features['x'] = X_tsne_3d[:, 0] 
    place_features['y'] = X_tsne_3d[:, 1]
    place_features['z'] = X_tsne_3d[:, 2]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(N_CLUSTERS):
        cluster_points = place_features[place_features['cluster'] == i]
        ax.scatter(cluster_points['x'], cluster_points['y'], cluster_points['z'], 
                   label=f'Cluster {i}', s=50)

    ax.set_title('3D Place Clustering (t-SNE)', fontsize=15)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.legend()
    
    plt.show()

# 막대그래프 시각화 함수 (모델 비교용)
def plot_metrics(metrics_df):
    if metrics_df is None:
        print("평가 지표 데이터가 없어 그래프를 생성할 수 없습니다.")
        return
        
    # 그래프 스타일 설정
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 3개의 지표를 위한 서브플롯 생성
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('클러스터링 모델 성능 비교', fontsize=16)

    # 지표별 정보 (제목, 색상 팔레트)
    plot_info = {
        'Silhouette': ('Silhouette Score (높을수록 좋음)', 'viridis'),
        'Davies-Bouldin': ('Davies-Bouldin Index (낮을수록 좋음)', 'plasma'),
        'Calinski-Harabasz': ('Calinski-Harabasz Index (높을수록 좋음)', 'magma')
    }

    # 각 지표에 대해 서브플롯 그리기
    for i, (metric, (title, palette)) in enumerate(plot_info.items()):
        # 데이터프레임을 지표 점수 기준으로 정렬
        df_sorted = metrics_df.sort_values(by=metric, ascending=False if '높을수록' in title else True)
        
        ax = axes[i]
        bars = sns.barplot(x=df_sorted.index, y=df_sorted[metric], ax=ax, palette=palette)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.tick_params(axis='x', rotation=10)

        # 막대 위에 수치 표시
        for bar in bars.patches:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{bar.get_height():.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 그래프를 이미지 파일로 저장
    plt.savefig("metrics_comparison.png", dpi=300)
    print("\n그래프가 'metrics_comparison.png' 파일로 저장되었습니다.")
    plt.show()

# 클러스터의 개수와 클러스터당 요소의 개수 확인
def print_cluster_counts():
    print("\n" + "="*50)
    print("      각 모델의 클러스터별 요소 개수 출력")
    print("="*50)

    try:
        kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
        dbscan = joblib.load(os.path.join(MODEL_DIR, 'dbscan.pkl'))
        agg_cluster = joblib.load(os.path.join(MODEL_DIR, 'agg_cluster.pkl'))
    except FileNotFoundError:
        print(f"\n오류: 모델 파일을 '{MODEL_DIR}' 폴더에서 찾을 수 없습니다.")
        print("이전 단계의 모델 학습 및 저장 스크립트를 먼저 실행해주세요.")
        return

    print("\n--- K-Means 클러스터별 요소 개수 ---")
    k_labels, k_counts = np.unique(kmeans.labels_, return_counts=True)
    kmeans_counts = dict(zip(k_labels, k_counts))
    for cluster_label, count in sorted(kmeans_counts.items()):
        print(f"  - 클러스터 {cluster_label}: {count} 개")

    print("\n--- DBSCAN 클러스터별 요소 개수 ---")
    d_labels, d_counts = np.unique(dbscan.labels_, return_counts=True)
    dbscan_counts = dict(zip(d_labels, d_counts))
    if -1 in dbscan_counts:
        print(f"  - 노이즈 (-1): {dbscan_counts[-1]} 개")
        del dbscan_counts[-1]

    for cluster_label, count in sorted(dbscan_counts.items()):
        print(f"  - 클러스터 {cluster_label}: {count} 개")

    print("\n--- 계층적 군집화(Agglomerative) 클러스터별 요소 개수 ---")
    a_labels, a_counts = np.unique(agg_cluster.labels_, return_counts=True)
    agg_counts = dict(zip(a_labels, a_counts))
    for cluster_label, count in sorted(agg_counts.items()):
        print(f"  - 클러스터 {cluster_label}: {count} 개")

if __name__ == '__main__':
    #train_save_models()
    #elbow_search()
    #grid_search()
    #metrics_dataframe = evaluation_metrics()
    #plot_metrics(metrics_dataframe)
    #print_cluster_counts()
    #visualize()
    visualize_3d()
    
