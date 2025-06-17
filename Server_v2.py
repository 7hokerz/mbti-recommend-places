import requests
import folium
import webbrowser
from openrouteservice import Client
import polyline
from itertools import permutations

# 🔑 API 키
KAKAO_API_KEY = '2086a74814fd03f298b88c0f30c1ea21'
ORS_API_KEY = '5b3ce3597851110001cf62485578ff6e26694c619b296e9a1dab6461'
client = Client(key=ORS_API_KEY)

# 📌 주소 → 좌표
def get_coordinates(address):
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}
    res = requests.get(url, headers=headers, params=params).json()
    if res['documents']:
        doc = res['documents'][0]
        return float(doc['x']), float(doc['y'])  # (lon, lat)
    return None

# 장소명으로 좌표 반환 (도로명 주소 아님)
def get_coordinates_from_place(place_name):
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": place_name}
    res = requests.get(url, headers=headers, params=params).json()
    if res['documents']:
        doc = res['documents'][0]
        return float(doc['x']), float(doc['y'])

# 🧭 ORS 경로 거리 및 좌표 반환
def get_route_geometry(coords):
    route = client.directions(coords, profile='driving-car')
    geometry = route['routes'][0]['geometry']
    distance = route['routes'][0]['summary']['distance']
    decoded = polyline.decode(geometry)
    coords_latlon = [(lat, lon) for lat, lon in decoded]
    return distance, coords_latlon

# 🍽️ 주변 장소 검색
def search_places(category, lon, lat, radius=1000):
    url = 'https://dapi.kakao.com/v2/local/search/category.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {
        "category_group_code": category,
        "x": lon, "y": lat,
        "radius": radius,
        "size": 15,
        "sort": "distance"
    }
    return requests.get(url, headers=headers, params=params).json()['documents']

# 🗺️ 지도 생성 (최적 경로만, 파랑→초록 구간 표시)
def create_map_segmented(start, mid, end, seg1_coords, seg2_coords, all_places):
    m = folium.Map(location=[start[1], start[0]], zoom_start=13)

    # 마커
    folium.Marker([start[1], start[0]], tooltip='출발지', icon=folium.Icon(color='blue')).add_to(m)
    folium.Marker([mid[1], mid[0]], tooltip='경유지', icon=folium.Icon(color='gray')).add_to(m)
    folium.Marker([end[1], end[0]], tooltip='도착지', icon=folium.Icon(color='red')).add_to(m)

    # 출발 → 경유 (파란색)
    folium.PolyLine(seg1_coords, color='blue', weight=6, tooltip="출발 → 경유").add_to(m)
    # 경유 → 도착 (초록색)
    folium.PolyLine(seg2_coords, color='green', weight=6, tooltip="경유 → 도착").add_to(m)

    # 주변 장소
    for place in all_places:
        lat, lon = float(place['y']), float(place['x'])
        name = place['place_name']
        cat = place['category_group_name']
        icon_color = 'green' if cat == '카페' else 'orange'
        folium.Marker([lat, lon], tooltip=name, icon=folium.Icon(color=icon_color)).add_to(m)

    return m

def process():
    print("장소명을 입력하세요 (예: 낙산공원, 혜화역 등)")
    names = [input(f"장소 {i+1}: ") for i in range(3)]

    #start_addr = input("출발지 주소를 입력하세요: ")
    #addr2 = input("두 번째 장소 주소를 입력하세요: ")
    #addr3 = input("세 번째 장소 주소를 입력하세요: ")

    start = get_coordinates_from_place(names[0])
    point2 = get_coordinates_from_place(names[1])
    point3 = get_coordinates_from_place(names[2])

    if not all([start, point2, point3]):
        print("❌ 좌표 변환 실패! 주소를 다시 확인하세요.")
        exit()

    print("✅ 출발지 좌표:", start)
    print("✅ 장소2 좌표:", point2)
    print("✅ 장소3 좌표:", point3)

    # 가능한 조합 생성
    candidates = [
        [start, p1, p2]
        for (p1, p2) in permutations([point2, point3])
    ]

    # 최단 경로 선택
    shortest_distance = float('inf')
    best_route = None

    for route in candidates:
        distance, _ = get_route_geometry(route)
        if distance < shortest_distance:
            shortest_distance = distance
            best_route = route

    # 최적 경로를 2구간으로 나눔
    _, seg1_coords = get_route_geometry([best_route[0], best_route[1]])
    _, seg2_coords = get_route_geometry([best_route[1], best_route[2]])

    # 장소 검색 (출발/경유/도착 + 중심지점 3곳)
    keypoints = best_route
    all_places = []
    for lon, lat in keypoints:
        all_places += search_places('CE7', lon, lat)
        all_places += search_places('FD6', lon, lat)

    # 지도 생성
    m = create_map_segmented(best_route[0], best_route[1], best_route[2], seg1_coords, seg2_coords, all_places)
    m.save("route_map_segmented_blue_green.html")
    print("최적 경로 지도 저장 완료: route_map_segmented_blue_green.html")
    webbrowser.open("route_map_segmented_blue_green.html")

# 🎯 실행
if __name__ == '__main__':
    print("장소명을 입력하세요 (예: 낙산공원, 혜화역 등)")
    names = [input(f"장소 {i+1}: ") for i in range(3)]

    #start_addr = input("출발지 주소를 입력하세요: ")
    #addr2 = input("두 번째 장소 주소를 입력하세요: ")
    #addr3 = input("세 번째 장소 주소를 입력하세요: ")

    start = get_coordinates_from_place(names[0])
    point2 = get_coordinates_from_place(names[1])
    point3 = get_coordinates_from_place(names[2])

    if not all([start, point2, point3]):
        print("❌ 좌표 변환 실패! 주소를 다시 확인하세요.")
        exit()

    print("✅ 출발지 좌표:", start)
    print("✅ 장소2 좌표:", point2)
    print("✅ 장소3 좌표:", point3)

    # 가능한 조합 생성
    candidates = [
        [start, p1, p2]
        for (p1, p2) in permutations([point2, point3])
    ]

    # 최단 경로 선택
    shortest_distance = float('inf')
    best_route = None

    for route in candidates:
        distance, _ = get_route_geometry(route)
        if distance < shortest_distance:
            shortest_distance = distance
            best_route = route

    # 최적 경로를 2구간으로 나눔
    _, seg1_coords = get_route_geometry([best_route[0], best_route[1]])
    _, seg2_coords = get_route_geometry([best_route[1], best_route[2]])

    # 장소 검색 (출발/경유/도착 + 중심지점 3곳)
    keypoints = best_route
    all_places = []
    for lon, lat in keypoints:
        all_places += search_places('CE7', lon, lat)
        all_places += search_places('FD6', lon, lat)

    # 지도 생성
    m = create_map_segmented(best_route[0], best_route[1], best_route[2], seg1_coords, seg2_coords, all_places)
    m.save("route_map_segmented_blue_green.html")
    print("최적 경로 지도 저장 완료: route_map_segmented_blue_green.html")
    webbrowser.open("route_map_segmented_blue_green.html")
