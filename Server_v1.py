import requests
import folium
import webbrowser
from openrouteservice import Client
import polyline

# Kakao REST API 키 (발급받은 키로 교체하세요)
KAKAO_API_KEY = '2086a74814fd03f298b88c0f30c1ea21'

# OpenRouteService API 키 (발급받은 키로 교체하세요)
ORS_API_KEY = '5b3ce3597851110001cf62485578ff6e26694c619b296e9a1dab6461'

# 📌 좌표 변환 (주소 → 좌표)
def get_coordinates(address):
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}
    res = requests.get(url, headers=headers, params=params).json()
    if res['documents']:
        doc = res['documents'][0]
        return float(doc['x']), float(doc['y'])
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

# 🧭 ORS 경로 계산
def get_route(start, end):
    client = Client(key=ORS_API_KEY)
    coords = [start, end]
    route = client.directions(coords, profile='driving-car', format='geojson')
    geometry = route['features'][0]['geometry']['coordinates']
    return geometry  # [ [lon, lat], [lon, lat], ...]

# 🍽️ 주변 장소 검색 (카카오)
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

# 🗺️ 지도 생성
def create_map(start, end, route_coords, nearby_places):
    m = folium.Map(location=[start[1], start[0]], zoom_start=13)

    # 출발/도착 마커
    folium.Marker([start[1], start[0]], tooltip='출발지', icon=folium.Icon(color='blue')).add_to(m)
    folium.Marker([end[1], end[0]], tooltip='도착지', icon=folium.Icon(color='red')).add_to(m)

    # 경로 라인
    route_latlon = [[lat, lon] for lon, lat in route_coords]
    folium.PolyLine(route_latlon, color='blue', weight=4).add_to(m)

    # 주변 장소 마커
    for place in nearby_places:
        lat, lon = float(place['y']), float(place['x'])
        name = place['place_name']
        cat = place['category_group_name']
        icon_color = 'green' if cat == '카페' else 'orange'
        folium.Marker([lat, lon], tooltip=name, icon=folium.Icon(color=icon_color)).add_to(m)

    return m

# 🎯 실행
if __name__ == '__main__':
    start_addr = input("출발지 장소명을 입력하세요: ")
    end_addr = input("도착지 장소명을 입력하세요: ")

    start = get_coordinates_from_place(start_addr)
    end = get_coordinates_from_place(end_addr)

    if not start or not end:
        print("❌ 좌표 변환 실패! 주소를 다시 확인하세요.")
        exit()

    print("✅ 출발지 좌표:", start)
    print("✅ 도착지 좌표:", end)

    route_coords = get_route(start, end)

    # 경로 중간 3지점 추출
    midpoints = [route_coords[len(route_coords)//4],
                 route_coords[len(route_coords)//2],
                 route_coords[3*len(route_coords)//4]]

    # 주변 장소 (카페 + 음식점)
    all_places = []
    for lon, lat in midpoints:
        all_places += search_places('CE7', lon, lat)  # 카페
        all_places += search_places('FD6', lon, lat)  # 음식점

    # 지도 만들기
    m = create_map(start, end, route_coords, all_places)
    m.save("route_map.html")
    print("📍 지도 저장 완료: route_map.html")

    webbrowser.open("route_map.html")
