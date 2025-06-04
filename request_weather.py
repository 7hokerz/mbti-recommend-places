import pandas as pd
import os
import datetime
import asyncio
import aiohttp

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

locationData = pd.read_csv(f"{script_dir}/data/location(weather).csv")

API_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
baseDate = datetime.date.today().strftime("%Y%m%d")
REQUEST_HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Pragma": "no-cache",
    "Sec-Ch-Ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": "\"Windows\"",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}

serviceKey = "Nkr+SDBQaBu8zVJgo/YYgInkXvle7ZbIDD1C2SWyUxsyXjzEOljWZ3LXoHAai5sIxOPf25UDnosYoLWVgzdx2Q=="

async def fetch_weather_for_row(session: aiohttp.ClientSession, sd: str, sgg: str):
    # 시도, 시군구에 맞는 X, Y 추출
    X = locationData[(locationData["1단계"] == sd) & (locationData["2단계"] == sgg)]["격자 X"].iloc[0]
    Y = locationData[(locationData["1단계"] == sd) & (locationData["2단계"] == sgg)]["격자 Y"].iloc[0]

    payload = {
        "serviceKey": serviceKey,
        "numOfRows": 1000,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": baseDate,
        "base_time": "0200", # 0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300
        "nx": X,
        "ny": Y,
    }

    for attempt in range(5): # 재시도 위한 반복문
        data_count = {"맑음": 0, "구름많음": 0, "흐림": 0, "없음": 0, "비": 0, "비/눈": 0, "눈": 0,"소나기": 0}
        try:
            async with session.get(API_URL, params=payload, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                response.raise_for_status() # HTTP 에러 발생 시 예외 발생
                weather_data = await response.json()

                if weather_data is None:
                    print("데이터를 가져오지 못했습니다. 재시도 중...")
                    if attempt < 4: await asyncio.sleep(1)
                    continue
            
            items_container = weather_data.get("response", {}).get("body", {}).get("items", {})
            status_items = items_container.get("item", [])

            if not status_items:
                print("응답에서 'item' 리스트를 찾을 수 없거나 비어있습니다.")
                header = weather_data.get("response", {}).get("header", {})
                if header:
                    print(f"API 응답 헤더: resultCode={header.get('resultCode')}, resultMsg={header.get('resultMsg')}")
                continue
        
            sky_data = [] # 하늘 상태
            pty_data = [] # 강수 형태

            for element in status_items:
                category = element.get("category")
                if category == 'SKY':
                    sky_data.append({
                        "날짜": element.get("fcstDate"),
                        "시간": element.get("fcstTime"),
                        "값": element.get("fcstValue"),
                    })
                elif category == 'PTY':
                    pty_data.append({
                        "날짜": element.get("fcstDate"),
                        "시간": element.get("fcstTime"),
                        "값": element.get("fcstValue"),
                    })
            
            #print("--- 하늘 상태 (SKY) ---")
            sky_code_meaning = {"1": "맑음", "3": "구름많음", "4": "흐림"}
            for e in sky_data:
                meaning = sky_code_meaning.get(e.get("값"), "알 수 없음")
                #print(f"날짜: {e.get('날짜')}, 시간: {e.get('시간')}, 값: {e.get('값')} ({meaning})")
                data_count[meaning] += 1

            #print("\n--- 강수 형태 (PTY) ---")
            pty_code_meaning = {"0": "없음", "1": "비", "2": "비/눈", "3": "눈","4": "소나기"}
            for e in pty_data:
                meaning = pty_code_meaning.get(e.get("값"), "알 수 없음")
                #print(f"날짜: {e.get('날짜')}, 시간: {e.get('시간e')}, 값: {e.get('값')} ({meaning})")
                data_count[meaning] += 1
            return data_count
            
        except KeyError as e:
            print(f"데이터 처리 중 키 오류 발생: 예상한 키 '{e}'가 응답에 없습니다.")
        except Exception as e:
            print(f"처리 중 알 수 없는 오류 발생: {e}")


async def get_weather_for_dataframe_async(df: pd.DataFrame):
    """DataFrame의 각 행에 대해 비동기적으로 날씨 정보를 가져옵니다."""
    tasks = []
    async with aiohttp.ClientSession(headers=REQUEST_HEADERS) as session:
        for _, row in df.iterrows(): # Using _ as index is unused
            task = fetch_weather_for_row(session, row['SIDO_NM'], row['SGG_NM'])
            tasks.append(task)
        
        # asyncio.gather를 사용하여 모든 작업을 동시에 실행하고 결과를 받음
        # return_exceptions=True로 설정하면 개별 작업에서 예외 발생 시 gather가 중단되지 않고 예외 객체를 결과 리스트에 포함
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return results

