"""
시연 장소 주변 도로망을 미리 넓게 받아서 graph_cache 폴더에 저장하는 스크립트.
GPS 아트 서버와 같은 폴더에서 실행한다. 서버가 켜져 있어도 되고, 저장된 파일은 다음 요청 때 자동으로 반영된다.

사용법:
  python prefetch_graph.py <위도> <경도> [반경m, 기본 2500]
  예) python prefetch_graph.py 37.5116 127.0595 2500

네트워크가 불안정하면 휴대폰 핫스팟 등 다른 네트워크에서 실행해도 된다. (받는 시간 동안만 연결되면 됨)
"""
import os
import sys
import time

import osmnx as ox

OVERPASS_URLS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]
GRAPH_CACHE_DIR = os.getenv("GPS_ART_GRAPH_CACHE_DIR", "graph_cache")

ox.settings.requests_timeout = 180   # 미리 받는 거라 넉넉하게 기다림
ox.settings.overpass_rate_limit = False


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    lat, lng = float(sys.argv[1]), float(sys.argv[2])
    radius = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

    for url in OVERPASS_URLS:
        ox.settings.overpass_url = url
        print(f"{url} 에서 반경 {radius}m 도로망 받는 중... (최대 3분)")
        start = time.time()
        try:
            G = ox.graph_from_point((lat, lng), dist=radius, network_type="walk", simplify=True)
        except Exception as e:
            print(f"  실패: {type(e).__name__} → 다음 서버 시도")
            continue
        path = os.path.join(GRAPH_CACHE_DIR, f"{lat:.4f}_{lng:.4f}_{radius}.graphml")
        ox.save_graphml(G, path)
        print(f"  완료 ({time.time() - start:.0f}초): 교차로 {G.number_of_nodes()}개 → {path}")
        return
    print("모든 서버에서 실패했습니다. 잠시 후 다시 시도하거나 다른 네트워크(핫스팟 등)에서 실행해 보세요.")
    sys.exit(1)


if __name__ == "__main__":
    main()