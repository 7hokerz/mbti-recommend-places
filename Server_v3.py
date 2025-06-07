import requests
import folium
import webbrowser
from openrouteservice import Client
import polyline
from itertools import permutations

# API 키
KAKAO_API_KEY = '2086a74814fd03f298b88c0f30c1ea21'
ORS_API_KEY = '5b3ce3597851110001cf62485578ff6e26694c619b296e9a1dab6461'
client = Client(key=ORS_API_KEY)

# 장소명으로 좌표 반환 (도로명 주소 아님)
def get_coordinates_from_place(place_name):
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": place_name}
    res = requests.get(url, headers=headers, params=params).json()
    if res['documents']:
        doc = res['documents'][0]
        return float(doc['x']), float(doc['y'])
    return None

# 경로 거리 계산
def get_total_distance(coords):
    route = client.directions(coords, profile='driving-car')
    distance = route['routes'][0]['summary']['distance']
    return distance

# 구간별 좌표 디코딩
def get_segment_route(p1, p2):
    route = client.directions([p1, p2], profile='driving-car')
    geometry = route['routes'][0]['geometry']
    decoded = polyline.decode(geometry)
    return [[lat, lon] for lat, lon in decoded]

# 지도 생성 함수
def create_map(loc_names, loc_coords):
    m = folium.Map(location=[loc_coords[0][1], loc_coords[0][0]], zoom_start=13)
    colors = ['blue', 'green']
    for i in range(2):
        segment = get_segment_route(loc_coords[i], loc_coords[i+1])
        folium.PolyLine(segment, color=colors[i], weight=6,
                        tooltip=f"{loc_names[i]} to {loc_names[i+1]}").add_to(m)

    for i, (name, coord) in enumerate(zip(loc_names, loc_coords)):
        color = 'blue' if i == 0 else ('gray' if i == 1 else 'red')
        folium.Marker([coord[1], coord[0]], tooltip=name,
                      icon=folium.Icon(color=color)).add_to(m)

    return m

# 실행부
if __name__ == '__main__':
    print("장소명을 입력하세요 (예: 낙산공원, 혜화역 등)")
    names = [input(f"장소 {i+1}: ") for i in range(3)]

    coords = []
    for name in names:
        coord = get_coordinates_from_place(name)
        if not coord:
            print(f"[오류] 장소명 '{name}'을(를) 찾을 수 없습니다.")
            exit()
        coords.append(coord)

    # 중간지점 순서 최적화
    best_order = None
    min_dist = float('inf')

    for perm in permutations(coords[1:]):
        path = [coords[0]] + list(perm)
        dist = get_total_distance(path)
        if dist < min_dist:
            min_dist = dist
            best_order = path
            best_names = [names[0]] + [names[1 + i] for i in [coords[1:].index(p) for p in perm]]

    # 지도 생성 및 저장
    m = create_map(best_names, best_order)
    m.save("optimal_map.html")
    print("최적 경로 지도 저장 완료: optimal_map.html")
    webbrowser.open("optimal_map.html")