import requests
import folium
from openrouteservice import Client

# Kakao REST API 키
KAKAO_API_KEY = '2086a74814fd03f298b88c0f30c1ea21'
# OpenRouteService API 키 
ORS_API_KEY = '5b3ce3597851110001cf62485578ff6e26694c619b296e9a1dab6461'

# 주소나 장소명을 좌표로 수정(is_address: True이면 주소, False이면 장소명)
def get_coordinates_unified(query, is_address=True):
    if isinstance(query, tuple) and len(query) == 2:
        return query
    
    if not isinstance(query, str):
        print(f"잘못된 입력: {query}")
        return None

    try:
        if is_address:
            url = f'https://dapi.kakao.com/v2/local/search/address.json?query={query}'
        else:
            url = f'https://dapi.kakao.com/v2/local/search/keyword.json?query={query}'
        headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        response = requests.get(url, headers=headers).json()
        documents = response.get('documents', [])
        if not documents:
            print(f"'{query}' → 좌표 변환 실패")
            return None
        x, y = float(documents[0]['x']), float(documents[0]['y'])
        return (x, y)
    except Exception as e:
        print(f"좌표 변환 중 오류: {e}")
        return None

# 경로 좌표 가져오기   
def get_route_coordinates(start_coords, end_coords):
    try:
        client = Client(key=ORS_API_KEY)
        coords = [start_coords, end_coords]
        route = client.directions(coords, profile='driving-car', format='geojson')
        geometry = route['features'][0]['geometry']['coordinates']
        return geometry
    except Exception as e:
        print(f"경로 좌표 가져오기 실패: {e}")
        return []

# 도로기준 거리 계산
def get_driving_distance(start_address: str, destination_name: str):
    start_coords = get_coordinates_unified(start_address, is_address=True)
    end_coords = get_coordinates_unified(destination_name, is_address=False)

    if not start_coords:
        print(f"출발 주소 '{start_address}' → 좌표 변환 실패")
        return None
    if not end_coords:
        print(f"도착 장소 '{destination_name}' → 좌표 변환 실패")
        return None

    try:
        client = Client(key=ORS_API_KEY)
        coords = [start_coords, end_coords]
        route = client.directions(coords, profile='driving-car')
        distance_meters = route['routes'][0]['summary']['distance']
        return round(distance_meters / 1000, 2)
    except Exception as e:
        print(f"거리 계산 실패 ({start_address} → {destination_name}) | 오류: {e}")
        return None
    


# 주변 장소 검색 
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

# 지도 생성
def create_map_kakao(start, end, route_coords, nearby_places):
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
