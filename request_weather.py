from dotenv import load_dotenv
import pandas as pd
import os
import datetime
import asyncio
import aiohttp

# 환경 변수 로드
load_dotenv()

# 현재 스크립트의 경로 및 위치 설정
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
locationData = pd.read_csv(f"{script_dir}/data/location(weather).csv") # 좌표 데이터 로딩

# 기상청 API
API_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
SERVICE_KEY = os.environ.get('WEATHER_API_KEY') # API 키
REQUEST_HEADERS = {
    "Accept": "*/*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
}

# 시도, 시군구 좌표 추출
def get_coords(sd: str, sgg: str):
    try:
        X = locationData[(locationData["1단계"] == sd) & (locationData["2단계"] == sgg)]["격자 X"].iloc[0]
        Y = locationData[(locationData["1단계"] == sd) & (locationData["2단계"] == sgg)]["격자 Y"].iloc[0]
        return X, Y
    except IndexError:
        print(f"[오류] 위치를 찾을 수 없습니다: {sd} {sgg}")
        return None, None

# 날씨 데이터에서 SKY, PTY 카운트 추출
def process_weather_items(items: list):
    if not items:
        return None
    
    # 날씨별 카운트
    data_count = {"맑음": 0, "구름많음": 0, "흐림": 0, "없음": 0, "비": 0, "비/눈": 0, "눈": 0,"소나기": 0}
    sky_code_meaning = {"1": "맑음", "3": "구름많음", "4": "흐림"}
    pty_code_meaning = {"0": "없음", "1": "비", "2": "비/눈", "3": "눈","4": "소나기"}
    
    # 각 항목에 분류 코드 및 카운트
    for item in items:
        category = item.get("category")
        value = item.get("fcstValue")
        if category == 'SKY':
            meaning = sky_code_meaning.get(value)
            data_count[meaning] += 1
        elif category == 'PTY':
            meaning = pty_code_meaning.get(value)
            data_count[meaning] += 1

    return data_count

# API 요청 함수
async def fetch_api_data(session: aiohttp.ClientSession, payload: dict):
    for attempt in range(5):
        try:
            async with session.get(API_URL, params=payload, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                response.raise_for_status() # HTTP 에러 발생 시 예외 발생
                weather_data = await response.json()

                header = weather_data.get("response", {}).get("header", {})
                if header.get("resultCode") != "00":
                    print(f"API 오류: {header.get('resultMsg')} (코드: {header.get('resultCode')})")
                    continue
                
                return weather_data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        except Exception as e:
            print(f"API 요청을 재시도 합니다. (시도 {attempt+1}/5): {e}")
            if attempt < 4:
                await asyncio.sleep(1)
    return None

# 한 행(장소)에 대한 날씨 요청
async def fetch_weather_for_row(session: aiohttp.ClientSession, sd: str, sgg: str):
    X, Y = get_coords(sd, sgg)
    if X is None:
        return None
    
    # 기상청 API 호출에 필요한 파라미터 구성
    payload = {
        "serviceKey": SERVICE_KEY,
        "numOfRows": 1000,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": datetime.date.today().strftime("%Y%m%d"),
        "base_time": "0200", # 0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300
        "nx": X,
        "ny": Y,
    }
    
    # API 호출 및 응답 데이터 추출
    items = await fetch_api_data(session, payload)
    
    # 응답이 없거나 실패한 경우 메시지 출력 후 None 반환
    if not items:  
        print(f"{sd} {sgg}의 날씨 정보를 가져오지 못했습니다.")
        return None  
            
    return process_weather_items(items)

# 여러 장소(데이터프레임)에 대한 날씨 요청
async def get_weather_for_dataframe_async(df: pd.DataFrame):
    tasks = []
    async with aiohttp.ClientSession(headers=REQUEST_HEADERS) as session:
        for _, row in df.iterrows():
            task = fetch_weather_for_row(session, row['SIDO_NM'], row['SGG_NM'])
            tasks.append(task)
        
        # asyncio.gather를 사용하여 모든 작업을 동시에 실행하고 결과를 받음
        # return_exceptions=True로 설정하면 개별 작업에서 예외 발생 시 gather가 중단되지 않고 예외 객체를 결과 리스트에 포함
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return results

# Test 
if __name__ == '__main__':
    async def main():
        sd = input("시/도를 입력하세요: ")
        sgg = input("시/군/구를 입력하세요: ")

        async with aiohttp.ClientSession(headers=REQUEST_HEADERS) as session:
            result = await fetch_weather_for_row(session, sd, sgg)

        if result:
            print("\n--- 날씨 정보 ---")
            print(result)
        else:
            print("날씨 정보를 가져오는 데 실패했습니다.")

    asyncio.run(main())
