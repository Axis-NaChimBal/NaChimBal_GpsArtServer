"""
GPS 아트 경로 생성 서버

사용자 위치 주변의 실제 보행자 도로망에서 폐곡선(순환 경로)을 찾고,
목표 도형(하트, 물고기, 네모, 십자가)과 가장 닮은 경로를 골라 반환한다.
하드코딩된 경로는 없으며, 주변에 폐곡선이 하나라도 있으면 항상 가장 비슷한 경로를 반환한다.

- POST /gps-art/generate
  요청: { "latitude": 37.5115, "longitude": 127.0595, "searchRadius": 1200, "shape": "square" }
        shape를 생략하면 하트 > 물고기 > 십자가 > 네모 순으로 기준을 통과한 도형을 고른다.
  응답(성공): { "found": true, "shape": "square", "fullPath": [...], "waypoints": [...],
               "totalDistanceMeters": 312.4, "distScore": 0.41, "exactMatch": true }
        exactMatch=false 이면 도형별 기준(SHAPE_THRESHOLDS)에는 못 미치지만 가장 비슷한 경로라는 뜻
  응답(실패): { "found": false }  ← 탐색 반경을 4배까지 넓혀도 닫힌 길이 하나도 없을 때만
- GET /gps-art/shapes : 지원 도형 목록 (heart, fish, square, cross)
- GET /gps-art/osm-status : OSM(Overpass) 서버별 접속 가능 여부·응답 속도
- POST /gps-art/osm-status/refresh : OSM 서버 상태 즉시 재점검
- GET /health

실행:
  pip install fastapi uvicorn osmnx networkx numpy shapely requests
  uvicorn gps_art_server:app --host 0.0.0.0 --port 8000
"""

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import networkx as nx
import numpy as np
import osmnx as ox
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from shapely.ops import substring

# ===== 도로망 그래프 캐시 =====
# OSM(Overpass) 공용 서버는 붐비면 수십 초~수 분씩 응답이 없을 때가 있다.
# 그래서 한 번 받은 도로망은 graph_cache 폴더에 저장해 두고, 요청 위치가 그 범위 안이면 다시 받지 않는다.
# (경로를 미리 정해두는 게 아니라 '도로 지도'만 재사용하는 것. 서버를 껐다 켜도 유지된다)
GRAPH_CACHE_DIR = os.getenv("GPS_ART_GRAPH_CACHE_DIR", "graph_cache")
os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

# Overpass 서버 목록. 실제로 쓸 때는 아래 '서버 상태 점검' 결과에 따라
# 살아 있는 서버를 빠른 순서로 먼저 쓰고, 죽은 서버는 건너뛴다.
OVERPASS_URLS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]
ox.settings.requests_timeout = 30        # 도로망 다운로드 시 서버 하나당 최대 대기(초)
ox.settings.overpass_rate_limit = False  # 미러 서버는 상태 확인 API가 없어서 끔

# ===== Overpass 서버 상태 점검 =====
# 서버를 켤 때 + 5분마다, 아주 작은 요청을 보내 응답하는 서버와 응답 속도를 기록해 둔다.
# 도로망을 받을 때는 '살아 있는 서버'만 빠른 순서로 시도하므로, 죽은 서버에서 25초씩 버리지 않는다.
HEALTH_CHECK_INTERVAL_SEC = 300
HEALTH_CHECK_TIMEOUT_SEC = 6
_HEALTH_QUERY = "[out:json][timeout:5];node(37.5,127.0,37.5005,127.0005);out 1;"
_server_status = {url: {"alive": None, "latency": None, "checked_at": None} for url in OVERPASS_URLS}
_status_lock = threading.Lock()


def _check_server(url):
    start = time.time()
    try:
        res = requests.post(
            url.rstrip("/") + "/interpreter",
            data={"data": _HEALTH_QUERY},
            headers={"User-Agent": "NaChimBal-GPS-Art/1.0"},
            timeout=HEALTH_CHECK_TIMEOUT_SEC,
        )
        alive = res.status_code == 200
    except Exception:
        alive = False
    latency = round(time.time() - start, 2)
    with _status_lock:
        _server_status[url] = {"alive": alive, "latency": latency if alive else None, "checked_at": time.time()}
    return url, alive, latency


def check_all_servers():
    with ThreadPoolExecutor(max_workers=len(OVERPASS_URLS)) as pool:
        results = list(pool.map(_check_server, OVERPASS_URLS))
    summary = ", ".join(f"{u.split('//')[1].split('/')[0]}={'OK ' + str(l) + 's' if a else 'X'}" for u, a, l in results)
    print(f"[OSM 점검] {summary}")


def _health_check_loop():
    while True:
        check_all_servers()
        time.sleep(HEALTH_CHECK_INTERVAL_SEC)


def servers_in_order():
    """살아 있는 서버(빠른 순) → 아직 점검 안 된 서버 → 죽은 서버(최후의 수단) 순서"""
    with _status_lock:
        status = dict(_server_status)
    alive = sorted((u for u in OVERPASS_URLS if status[u]["alive"]), key=lambda u: status[u]["latency"])
    unknown = [u for u in OVERPASS_URLS if status[u]["alive"] is None]
    dead = [u for u in OVERPASS_URLS if status[u]["alive"] is False]
    if alive:
        return alive + unknown  # 살아 있는 서버가 있으면 죽은 서버는 아예 시도하지 않음
    return unknown + dead       # 전부 죽었다고 나오면 그래도 한 번씩은 시도


def reset_server_status():
    with _status_lock:
        for url in OVERPASS_URLS:
            _server_status[url] = {"alive": None, "latency": None, "checked_at": None}


def mark_dead(url):
    with _status_lock:
        _server_status[url] = {"alive": False, "latency": None, "checked_at": time.time()}


threading.Thread(target=_health_check_loop, daemon=True).start()

_graph_cache = []  # [(bbox, graph)], bbox = (min_lat, max_lat, min_lng, max_lng)


def _graph_bbox(G):
    ys = [d["y"] for _, d in G.nodes(data=True)]
    xs = [d["x"] for _, d in G.nodes(data=True)]
    return (min(ys), max(ys), min(xs), max(xs))


_loaded_cache_files = set()


def refresh_cache_from_disk():
    """graph_cache 폴더에 새로 생긴 파일(prefetch_graph.py로 받은 것 등)을 읽어 들인다."""
    added = 0
    for name in sorted(os.listdir(GRAPH_CACHE_DIR)):
        if not name.endswith(".graphml") or name in _loaded_cache_files:
            continue
        _loaded_cache_files.add(name)
        try:
            G = ox.load_graphml(os.path.join(GRAPH_CACHE_DIR, name))
            _graph_cache.append((_graph_bbox(G), G))
            added += 1
        except Exception as e:
            print(f"[캐시] {name} 로드 실패: {e}")
    return added


refresh_cache_from_disk()
print(f"[캐시] 저장된 도로망 {len(_graph_cache)}개 로드 ({GRAPH_CACHE_DIR} 폴더)")

app = FastAPI(title="NaChimBal GPS Art Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 알고리즘 설정값 =====
CYCLE_LENGTH_MIN = 0          # 폐곡선 후보 최소 둘레(m). 0 = 제한 없음
CYCLE_LENGTH_MAX = 20000      # 폐곡선 후보 최대 둘레(m)
MAX_FACE_NODES = 60           # 블록(면) 하나를 이루는 최대 교차로 수
MAX_CANDIDATES = 400          # 사용자와 가까운 순으로 이만큼만 도형 비교 (속도용)
SIMPLIFY_MIN_M = 3.0          # 윤곽 단순화 최소 허용 오차(m)
SIMPLIFY_RATIO = 0.02         # 윤곽 단순화 허용 오차 = 둘레 × 이 비율
RADIUS_EXPAND_STEPS = (1, 2, 4)  # 후보가 없으면 탐색 반경을 이 배수로 넓혀 재시도
RATIO_WEIGHT = 20             # 변 길이 비율 DTW 가중치
MAX_VERTEX_RATIO = 2.5        # 목표 도형 대비 꼭짓점 개수 허용 배수 (초과분은 페널티)
VERTEX_PENALTY_WEIGHT = 2.0
SIMILARITY_THRESHOLD = 2.0    # (참고용 기본값) 도형별 합격 점수는 SHAPE_THRESHOLDS 사용

# shape를 지정하지 않았을 때의 도형 우선순위.
# 앞 도형부터 기준(SHAPE_THRESHOLDS)을 통과한 경로가 있으면 그 도형을 고른다.
SHAPE_PRIORITY = ["heart", "fish", "cross", "square"]

# 도형별 합격 점수 (낮을수록 엄격). 하트로 보기 애매한 경로가 하트로 뽑히면 이 값을 낮춘다.
# uvicorn 창의 [결과] 로그에서 "하트 같은 것"과 "아닌 것"의 점수를 보고 그 사이 값으로 맞추면 된다.
SHAPE_THRESHOLDS = {
    "heart": 1.5,
    "fish": 1.8,
    "cross": 2.0,
    "square": 2.0,
}

# 도형별로 반드시 있어야 하는 '안으로 꺾인 모서리' 개수.
# 하트는 위쪽 가운데가 안으로 파여 있어야 하고(1개), 물고기는 꼬리 연결부(2개), 십자가는 안쪽 모서리 4개.
# 이게 없으면 점수가 좋아도 그 도형으로 인정하지 않는다. (둥글둥글한 덩어리가 하트로 뽑히는 것 방지)
MIN_CONCAVE_CORNERS = {
    "heart": 1,
    "fish": 2,
    "cross": 4,
    "square": 0,
}
CONCAVE_ANGLE_RAD = math.radians(25)  # 이 각도 이상 반대로 꺾여야 '안으로 꺾인 모서리'로 셈


# ===== 목표 도형 (0~1 정규화 좌표, 시작점으로 닫힌 형태) =====
SHAPES = {
    "heart": np.array([
        [0.5, 1.0], [0.2, 0.7], [0.0, 0.4], [0.2, 0.1],
        [0.5, 0.3], [0.8, 0.1], [1.0, 0.4], [0.8, 0.7], [0.5, 1.0],
    ]),
    "fish": np.array([
        [0.0, 0.5], [0.25, 0.8], [0.55, 0.8], [0.75, 0.55], [1.0, 0.85],
        [1.0, 0.15], [0.75, 0.45], [0.55, 0.2], [0.25, 0.2], [0.0, 0.5],
    ]),
    "square": np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],
    ]),
    "cross": np.array([
        [1/3, 1.0], [2/3, 1.0], [2/3, 2/3], [1.0, 2/3],
        [1.0, 1/3], [2/3, 1/3], [2/3, 0.0], [1/3, 0.0],
        [1/3, 1/3], [0.0, 1/3], [0.0, 2/3], [1/3, 2/3], [1/3, 1.0],
    ]),
}


# ===== 요청/응답 모델 =====
class GpsArtRequest(BaseModel):
    latitude: float
    longitude: float
    searchRadius: int = 1200
    shape: str | None = None


class GpsArtResponse(BaseModel):
    found: bool
    shape: str | None = None
    fullPath: list | None = None
    waypoints: list | None = None
    totalDistanceMeters: float | None = None
    distScore: float | None = None
    exactMatch: bool | None = None


# ===== 좌표 변환 =====
def meter_scale(lat0):
    """경위도(도) → 미터 변환 계수. 위도 37도에서 경도 1도는 위도 1도보다 약 20% 짧다."""
    return 111320 * math.cos(math.radians(lat0)), 110540


def to_meters(points, lat0):
    kx, ky = meter_scale(lat0)
    return [(x * kx, y * ky) for x, y in points]


# ===== 폐곡선 처리 =====
def cycle_length(cycle, G_multi):
    total = 0
    for i in range(len(cycle)):
        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
        if G_multi.has_edge(u, v):
            edge_data = G_multi.get_edge_data(u, v)
            first_key = list(edge_data.keys())[0]
            total += edge_data[first_key].get("length", 0)
    return total


def cycle_to_coords(cycle, G):
    return [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in cycle]


def cycle_to_detailed_coords(cycle, G_multi):
    """노드 사이의 실제 도로 곡선까지 포함한 좌표"""
    detailed = []
    for i in range(len(cycle)):
        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
        detailed.append((G_multi.nodes[u]["x"], G_multi.nodes[u]["y"]))
        if G_multi.has_edge(u, v):
            edge_data = G_multi.get_edge_data(u, v)
            first_key = list(edge_data.keys())[0]
            edge = edge_data[first_key]
            if "geometry" in edge:
                xs, ys = edge["geometry"].xy
                points = list(zip(xs, ys))
                # geometry가 v→u 방향으로 저장된 경우 뒤집어서 경로가 끊기지 않게 함
                u_pt = (G_multi.nodes[u]["x"], G_multi.nodes[u]["y"])
                if points and _sq_dist(points[-1], u_pt) < _sq_dist(points[0], u_pt):
                    points.reverse()
                detailed.extend(points)

    # 노드 좌표와 도로 곡선의 끝점이 겹치는 경우 연속 중복 좌표 제거
    deduped = []
    for p in detailed:
        p = (float(p[0]), float(p[1]))
        if not deduped or deduped[-1] != p:
            deduped.append(p)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def _sq_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _rdp(points, tol):
    """Douglas-Peucker: 양 끝점은 고정하고, 허용 오차 이내의 자잘한 굴곡 점을 지운다."""
    if len(points) < 3:
        return points
    a, b = np.array(points[0]), np.array(points[-1])
    mid = np.array(points[1:-1])
    ab = b - a
    ab_len = math.hypot(ab[0], ab[1])
    if ab_len == 0:
        d = np.hypot(mid[:, 0] - a[0], mid[:, 1] - a[1])
    else:
        d = np.abs(ab[0] * (mid[:, 1] - a[1]) - ab[1] * (mid[:, 0] - a[0])) / ab_len
    i = int(np.argmax(d))
    if d[i] <= tol:
        return [points[0], points[-1]]
    return _rdp(points[: i + 2], tol)[:-1] + _rdp(points[i + 1:], tol)


def simplify_ring(xy, tol):
    """
    닫힌 윤곽선 단순화. shapely의 simplify는 닫힌 선의 시작점을 지워버려서
    네모의 모서리 하나가 사라지고 세모가 되는 문제가 있었다.
    → 중심에서 가장 먼 점과, 그 점에서 가장 먼 점(둘 다 확실한 꼭짓점)으로 반씩 나눠 처리한다.
    """
    arr = np.array(xy)
    center = arr.mean(axis=0)
    i0 = int(np.argmax(np.hypot(arr[:, 0] - center[0], arr[:, 1] - center[1])))
    i1 = int(np.argmax(np.hypot(arr[:, 0] - arr[i0, 0], arr[:, 1] - arr[i0, 1])))
    if i0 == i1:
        return xy
    ring = xy[i0:] + xy[:i0]
    j = (i1 - i0) % len(xy)
    first = _rdp(ring[: j + 1], tol)
    second = _rdp(ring[j:] + [ring[0]], tol)
    return first[:-1] + second[:-1]


def outline_for_matching(cycle, G_multi, lat0):
    """
    도형 비교용 윤곽선.
    교차로 노드만 쓰면 '교차로가 아닌 꺾인 모서리'가 사라져서 네모 블록이 세모·선분처럼 보인다.
    그래서 실제 도로 곡선까지 펼친 뒤(미터 좌표), 자잘한 굴곡만 단순화해서 꼭짓점을 되살린다.
    """
    xy = to_meters(cycle_to_detailed_coords(cycle, G_multi), lat0)
    if len(xy) < 3:
        return None
    perimeter = sum(math.dist(xy[i], xy[(i + 1) % len(xy)]) for i in range(len(xy)))
    tol = max(SIMPLIFY_MIN_M, perimeter * SIMPLIFY_RATIO)
    simplified = simplify_ring(xy, tol)
    return simplified if len(simplified) >= 3 else xy


# ===== 형태 시그니처 & 유사도 =====
def shape_signature(coords):
    """회전각 시퀀스와 변 길이 비율 시퀀스 (위치·크기·회전에 무관한 도형 특징)"""
    coords = np.array(coords, dtype=float)
    # 시작점으로 닫힌 좌표면 마지막 중복 점 제거 (길이 0짜리 변이 생기지 않게)
    if len(coords) > 1 and np.linalg.norm(coords[0] - coords[-1]) < 1e-9:
        coords = coords[:-1]
    if len(coords) < 3:
        return None, None

    vecs = np.roll(coords, -1, axis=0) - coords
    seg_lengths = np.linalg.norm(vecs, axis=1)
    seg_lengths[seg_lengths == 0] = 1e-9
    length_ratios = seg_lengths / seg_lengths.sum()

    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    turning_angles = np.diff(angles, append=angles[0])
    turning_angles = (turning_angles + np.pi) % (2 * np.pi) - np.pi

    return turning_angles, length_ratios


def dtw_distance(seq_a, seq_b, circular=False):
    n, m = len(seq_a), len(seq_b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diff = abs(seq_a[i - 1] - seq_b[j - 1])
            cost = min(diff, 2 * np.pi - diff) if circular else diff
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw[n, m]


def signature_distance(cand_angles, cand_ratios, ref_angles, ref_ratios):
    d_angle = dtw_distance(cand_angles, ref_angles, circular=True) / len(ref_angles)
    d_ratio = dtw_distance(cand_ratios, ref_ratios, circular=False) / len(ref_ratios)
    return d_angle + d_ratio * RATIO_WEIGHT


def vertex_penalty(n_cand, n_ref):
    return max(0, (n_cand / n_ref) - MAX_VERTEX_RATIO) * VERTEX_PENALTY_WEIGHT


def best_match_score(cand_angles, cand_ratios, ref_angles, ref_ratios):
    """후보 폐곡선의 시작점·진행 방향을 모두 돌려보며 목표 도형과의 최소 거리를 구함"""
    penalty = vertex_penalty(len(cand_angles), len(ref_angles))

    # 반대 방향으로 돌면 변 순서가 뒤집히고, 회전각은 부호가 바뀐다(좌회전 ↔ 우회전).
    # (기존 코드는 순서만 뒤집고 부호를 안 바꿔서, 반대로 도는 네모는 매번 큰 점수를 받았음)
    reversed_angles = -np.roll(cand_angles[::-1], -1)
    reversed_ratios = cand_ratios[::-1]

    best = float("inf")
    for angles, ratios in ((cand_angles, cand_ratios), (reversed_angles, reversed_ratios)):
        for start in range(len(angles)):
            best = min(best, signature_distance(
                np.roll(angles, -start), np.roll(ratios, -start), ref_angles, ref_ratios
            ))

    return best + penalty


SHAPE_SIGNATURES = {name: shape_signature(coords) for name, coords in SHAPES.items()}


def count_concave_corners(turning_angles):
    """
    안으로 꺾인 모서리 개수.
    한 바퀴 도는 방향(시계/반시계)과 반대로 크게 꺾이는 모서리가 '안으로 파인' 곳이다.
    """
    direction = 1 if turning_angles.sum() > 0 else -1
    return int(np.sum(turning_angles * direction < -CONCAVE_ANGLE_RAD))


# ===== 도로망 그래프 =====
def build_deterministic_simple_graph(G_undirected):
    """
    노드/엣지를 정렬된 순서로 넣어, 그래프를 새로 받아오든 캐시에서 읽든
    탐색 결과가 항상 같게 나오도록 고정한다.
    """
    G_simple = nx.Graph()
    for n in sorted(G_undirected.nodes()):
        G_simple.add_node(n)
    for u, v in sorted(G_undirected.edges()):
        if u != v:  # 자기 자신으로 돌아오는 엣지 제외
            G_simple.add_edge(u, v)
    return G_simple


def split_parallel_and_loop_edges(G_undirected):
    """
    두 교차로 사이에 길이 두 갈래인 경우(교차로 2개짜리 네모 블록)와
    한 교차로에서 나가서 다시 돌아오는 길(작은 순환로)은 단순 그래프로 바꾸는 순간 사라진다.
    이런 길 중간에 가상 지점을 넣어 일반 고리로 만들어 둔다. (가상 지점도 실제 도로 위의 점)
    """
    G = G_undirected.copy()
    next_id = -1
    for u, v, k, data in list(G_undirected.edges(keys=True, data=True)):
        if u == v:
            parts = 3
        elif k != min(G_undirected[u][v]):
            parts = 2
        else:
            continue
        geom = data.get("geometry")
        if geom is None or geom.length == 0:
            continue  # 모양 정보 없는 중복 직선은 같은 길이라 버림

        start_pt = geom.coords[0]
        u_pt = (G.nodes[u]["x"], G.nodes[u]["y"])
        v_pt = (G.nodes[v]["x"], G.nodes[v]["y"])
        a, b = (u, v) if _sq_dist(start_pt, u_pt) <= _sq_dist(start_pt, v_pt) else (v, u)

        chain = [a]
        for i in range(1, parts):
            p = geom.interpolate(i / parts, normalized=True)
            G.add_node(next_id, x=p.x, y=p.y)
            chain.append(next_id)
            next_id -= 1
        chain.append(b)

        G.remove_edge(u, v, k)
        for i in range(parts):
            attrs = dict(data)
            attrs["length"] = data.get("length", 0) / parts
            attrs["geometry"] = substring(geom, i / parts, (i + 1) / parts, normalized=True)
            G.add_edge(chain[i], chain[i + 1], **attrs)
    return G


def enumerate_blocks(G_simple, pos):
    """
    지도에서 눈에 보이는 '블록'(도로로 둘러싸인 가장 작은 칸)을 전부 찾는다.
    각 교차로에서 들어온 길 기준으로 항상 바로 옆 길로 꺾으며 한 바퀴 돌면 블록 하나가 나온다.
    cycle_basis()는 블록이 아니라 여러 블록이 합쳐진 큰 고리를 주는 경우가 많아서,
    작은 네모 블록이 후보에서 아예 빠지는 원인이었다.
    """
    core = nx.k_core(G_simple, 2)  # 막다른 길(고리를 만들 수 없는 가지) 제거

    order, index = {}, {}
    for v in core.nodes():
        vx, vy = pos[v]
        nbrs = sorted(core.neighbors(v), key=lambda w: math.atan2(pos[w][1] - vy, pos[w][0] - vx))
        order[v] = nbrs
        index[v] = {w: i for i, w in enumerate(nbrs)}

    visited = set()
    blocks = []
    for u, v in core.edges():
        for a, b in ((u, v), (v, u)):
            if (a, b) in visited:
                continue
            face = []
            x, y = a, b
            while (x, y) not in visited and len(face) <= MAX_FACE_NODES:
                visited.add((x, y))
                face.append(x)
                nbrs = order[y]
                w = nbrs[(index[y][x] - 1) % len(nbrs)]
                x, y = y, w
            if (x, y) != (a, b):              # 너무 커서 중간에 끊긴 면
                continue
            if len(face) < 3 or len(set(face)) != len(face):  # 같은 교차로를 두 번 지나는 면
                continue
            if _signed_area([pos[n] for n in face]) <= 0:      # 바깥 테두리(외곽 면) 제외
                continue
            blocks.append(face)
    return blocks


def _signed_area(points):
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2


def _request_box(latitude, longitude, search_radius):
    kx, ky = meter_scale(latitude)
    r = search_radius * 0.9  # 가장자리 교차로가 반경보다 살짝 안쪽에 있는 것 감안
    return latitude - r / ky, latitude + r / ky, longitude - r / kx, longitude + r / kx


def _inside(bbox, lat, lng):
    min_lat, max_lat, min_lng, max_lng = bbox
    return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng


def _overlaps(bbox, box):
    return not (bbox[1] < box[0] or bbox[0] > box[1] or bbox[3] < box[2] or bbox[2] > box[3])


def _cached_graphs_for(latitude, longitude, search_radius, require_full_cover=True):
    """
    요청 범위와 겹치는 저장된 도로망들을 찾는다.
    여러 파일에 걸쳐 있어도 합쳐서 요청 범위를 다 덮으면 사용할 수 있다.
    (OSM 교차로 ID는 어느 파일에서나 같아서, 겹치는 부분은 같은 교차로로 자연스럽게 이어진다)
    """
    box = _request_box(latitude, longitude, search_radius)
    graphs = [(bbox, G) for bbox, G in _graph_cache if _overlaps(bbox, box)]
    if not graphs:
        return []
    if require_full_cover:
        # 요청 범위 안에 5x5 점을 찍어서, 모든 점이 어느 한 파일 범위 안에 들어가면 '다 덮음'
        for i in range(5):
            for j in range(5):
                lat = box[0] + (box[1] - box[0]) * i / 4
                lng = box[2] + (box[3] - box[2]) * j / 4
                if not any(_inside(bbox, lat, lng) for bbox, _ in graphs):
                    return []
    return [G for _, G in graphs]


def _merge(graphs):
    return graphs[0] if len(graphs) == 1 else nx.compose_all(graphs)


def _download_graph(latitude, longitude, search_radius):
    last_error = None
    for url in servers_in_order():
        ox.settings.overpass_url = url
        try:
            print(f"[OSM] {url} 에서 도로망 다운로드 (반경 {search_radius}m)")
            return ox.graph_from_point(
                (latitude, longitude), dist=search_radius, network_type="walk", simplify=True
            )
        except Exception as e:
            print(f"[OSM] {url} 실패: {type(e).__name__} → 다음 점검 전까지 이 서버는 건너뜀")
            mark_dead(url)
            last_error = e
    raise last_error


def _crop_graph(G, latitude, longitude, search_radius):
    """넓게 저장된 도로망에서 요청 위치 주변 search_radius 안쪽만 잘라서 계산에 사용"""
    kx, ky = meter_scale(latitude)
    keep = [
        n for n, d in G.nodes(data=True)
        if math.hypot((d["x"] - longitude) * kx, (d["y"] - latitude) * ky) <= search_radius
    ]
    return G.subgraph(keep).copy()

def load_graph(latitude, longitude, search_radius, allow_cache=True):
    """
    ① 미리 저장해 둔 도로망이 요청 범위를 덮으면 그것 사용 (OSM 접속 안 함)
    ② 아니면 OSM에서 요청 반경만큼만 받아서 바로 사용 (저장 안 함)
    ③ OSM도 실패하면 일부라도 겹치는 저장된 도로망으로 탐색
    """
    refresh_cache_from_disk()

    graphs = _cached_graphs_for(latitude, longitude, search_radius)
    if graphs:
        print(f"[캐시] 저장된 도로망 {len(graphs)}개로 범위를 덮음 → OSM 접속 안 함")
        return _crop_graph(_merge(graphs), latitude, longitude, search_radius)

    partial = _cached_graphs_for(latitude, longitude, search_radius, require_full_cover=False)
    partial_graph = _crop_graph(_merge(partial), latitude, longitude, search_radius) if partial else None
    has_partial = partial_graph is not None and partial_graph.number_of_nodes() > 0

    # 점검 결과 OSM 서버가 전부 죽어 있으면 기다리지 않고 바로 저장된 도로망 사용
    with _status_lock:
        all_dead = all(st["alive"] is False for st in _server_status.values())
    if all_dead and has_partial:
        print("[캐시] OSM 서버 전부 응답 없음 → 일부만 겹치는 저장된 도로망으로 바로 탐색")
        return partial_graph

    print(f"[OSM] 저장 구역 밖 → 반경 {search_radius}m 받아서 바로 사용 (저장 안 함)")
    try:
        return _download_graph(latitude, longitude, search_radius)
    except Exception:
        if has_partial:
            print("[캐시] OSM 연결 실패 → 일부만 겹치는 저장된 도로망으로 대신 탐색")
            return partial_graph
        raise

# ===== 후보 수집 =====
def collect_candidates(G, latitude, longitude):
    G_undirected = split_parallel_and_loop_edges(ox.convert.to_undirected(G))
    G_simple = build_deterministic_simple_graph(G_undirected)
    pos = dict(zip(G_simple.nodes(), to_meters(cycle_to_coords(list(G_simple.nodes()), G_undirected), latitude)))

    # 작은 블록(면) + 여러 블록을 합친 큰 고리(하트·물고기용)를 모두 후보로
    seen = set()
    cycles = []
    for cycle in enumerate_blocks(G_simple, pos) + nx.cycle_basis(G_simple):
        key = frozenset(cycle)
        if len(cycle) >= 3 and key not in seen:
            seen.add(key)
            cycles.append(cycle)

    # 사용자와 가까운 고리부터 비교
    ux, uy = to_meters([(longitude, latitude)], latitude)[0]

    def dist_to_user(cycle):
        cx = sum(pos[n][0] for n in cycle) / len(cycle)
        cy = sum(pos[n][1] for n in cycle) / len(cycle)
        return math.hypot(cx - ux, cy - uy)

    cycles.sort(key=dist_to_user)

    candidates = []
    for cycle in cycles:
        length = cycle_length(cycle, G_undirected)
        if not (CYCLE_LENGTH_MIN <= length <= CYCLE_LENGTH_MAX):
            continue
        outline = outline_for_matching(cycle, G_undirected, latitude)
        if outline is None:
            continue
        angles, ratios = shape_signature(outline)
        if angles is None:
            continue
        candidates.append((cycle, length, angles, ratios))
        if len(candidates) >= MAX_CANDIDATES:
            break

    return G_undirected, candidates


# ===== GPS 아트 탐색 =====
def find_best_gps_art(latitude: float, longitude: float, search_radius: int, shape: str | None = None):
    """
    중심 좌표 주변에서 목표 도형과 가장 닮은 폐곡선을 찾는다.
    기준 점수를 못 넘어도 가장 비슷한 경로를 반환하고, 닫힌 길이 아예 없을 때만 None.
    """
    target_shapes = [shape] if shape else SHAPE_PRIORITY

    t_start = time.time()
    G_undirected, candidates = None, []
    for i, factor in enumerate(RADIUS_EXPAND_STEPS):
        t0 = time.time()
        G = load_graph(latitude, longitude, search_radius * factor, allow_cache=(i == 0))
        t1 = time.time()
        G_undirected, candidates = collect_candidates(G, latitude, longitude)
        print(f"[시간] 도로망 준비 {t1 - t0:.1f}s, 후보 {len(candidates)}개 수집 {time.time() - t1:.1f}s")
        if candidates:
            break
        print(f"[결과] 반경 {search_radius * factor}m 안에 닫힌 길 없음 → 반경을 넓혀 다시 탐색")

    if not candidates:
        return None

    # 도형별로 가장 닮은 경로를 하나씩 구함
    best_by_shape = {}  # shape_name -> (score, shape_name, cycle, length)
    for shape_name in target_shapes:
        ref_angles, ref_ratios = SHAPE_SIGNATURES[shape_name]
        for cycle, length, angles, ratios in candidates:
            # 그 도형에 꼭 필요한 '파인 모서리'가 없으면 이 도형 후보에서 제외
            if count_concave_corners(angles) < MIN_CONCAVE_CORNERS[shape_name]:
                continue
            score = best_match_score(angles, ratios, ref_angles, ref_ratios)
            current = best_by_shape.get(shape_name)
            if current is None or score < current[0]:
                best_by_shape[shape_name] = (score, shape_name, cycle, length)

    # 우선순위(하트 > 물고기 > 십자가 > 네모) 순서로 기준을 통과한 첫 도형을 선택.
    # 어느 도형도 기준을 못 넘으면 점수가 가장 좋은 경로를 반환.
    best = next(
        (best_by_shape[name] for name in target_shapes
         if name in best_by_shape and best_by_shape[name][0] <= SHAPE_THRESHOLDS[name]),
        None,
    )
    if best is None and best_by_shape:
        best = min(best_by_shape.values(), key=lambda b: b[0])
    if best is None:
        # 지정한 도형의 조건(파인 모서리)을 만족하는 경로가 하나도 없으면, 조건 없이 가장 비슷한 경로
        ref_angles, ref_ratios = SHAPE_SIGNATURES[target_shapes[0]]
        best = min(
            ((best_match_score(a, r, ref_angles, ref_ratios), target_shapes[0], c, l) for c, l, a, r in candidates),
            key=lambda b: b[0],
        )

    best_score, best_shape, best_cycle, best_length = best
    print(f"[시간] 전체 {time.time() - t_start:.1f}s")

    detailed_coords = cycle_to_detailed_coords(best_cycle, G_undirected)
    detailed_coords.append(detailed_coords[0])

    return {
        "shape": best_shape,
        "fullPath": [{"x": x, "y": y} for x, y in detailed_coords],
        "waypoints": [
            {"latitude": float(lat), "longitude": float(lng)}
            for lng, lat in cycle_to_coords(best_cycle, G_undirected)
        ],
        "totalDistanceMeters": round(best_length, 1),
        "distScore": round(float(best_score), 4),
        "exactMatch": bool(best_score <= SHAPE_THRESHOLDS[best_shape]),
    }


# ===== API =====
@app.post("/gps-art/generate", response_model=GpsArtResponse)
def generate_gps_art(req: GpsArtRequest):
    print(f"[요청] lat={req.latitude}, lng={req.longitude}, radius={req.searchRadius}, shape={req.shape}")
    if req.shape is not None and req.shape not in SHAPES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 도형입니다: {req.shape} (지원: {', '.join(SHAPES)})",
        )

    try:
        result = find_best_gps_art(req.latitude, req.longitude, req.searchRadius, req.shape)
    except Exception as e:
        print(f"[결과] 도로망을 받아오지 못함: {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail="OSM 도로망 서버에 연결하지 못했습니다. 잠시 후 다시 시도해주세요.")
    if result is None:
        print("[결과] 경로 없음")
        return GpsArtResponse(found=False)

    print(f"[결과] 도형={result['shape']}, 점수={result['distScore']}, "
          f"기준통과={result['exactMatch']}, 거리={result['totalDistanceMeters']}m")
    return GpsArtResponse(found=True, **result)


@app.get("/gps-art/shapes")
def list_shapes():
    return {"shapes": list(SHAPES.keys())}


@app.get("/gps-art/osm-status")
def osm_status():
    """Overpass 서버 상태 확인용 (브라우저에서 열어보면 됨)"""
    with _status_lock:
        status = dict(_server_status)
    return {
        "order": servers_in_order(),
        "servers": {
            url: {
                "alive": st["alive"],
                "latencySec": st["latency"],
                "checkedSecAgo": None if st["checked_at"] is None else round(time.time() - st["checked_at"]),
            }
            for url, st in status.items()
        },
    }


@app.post("/gps-art/osm-status/refresh")
def osm_status_refresh():
    """지금 바로 다시 점검"""
    check_all_servers()
    return osm_status()


@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))   # 환경변수 PORT가 없으면 8001
    uvicorn.run(app, host="0.0.0.0", port=port)