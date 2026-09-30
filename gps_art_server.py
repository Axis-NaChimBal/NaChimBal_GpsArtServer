"""
GPS 아트 경로 생성 서버

사용자 위치 주변의 실제 보행자 도로망에서 폐곡선(순환 경로)을 찾고,
목표 도형(하트, 물고기, 별 등)과 가장 닮은 경로를 골라 반환한다.

- POST /gps-art/generate
  요청: { "latitude": 37.5115, "longitude": 127.0595, "searchRadius": 1200, "shape": "heart" }
        shape를 생략하면 지원하는 모든 도형 중 가장 잘 맞는 도형을 자동으로 선택한다.
  응답(성공): { "found": true, "shape": "heart", "fullPath": [...], "waypoints": [...],
               "totalDistanceMeters": 806.0, "distScore": 1.225 }
  응답(실패): { "found": false }
- GET /gps-art/shapes : 지원 도형 목록
- GET /health

실행:
  pip install fastapi uvicorn osmnx networkx numpy shapely
  uvicorn gps_art_server:app --host 0.0.0.0 --port 8000
"""

import os

import networkx as nx
import numpy as np
import osmnx as ox
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===== 도로망 그래프 캐시 =====
# prefetch_graph.py로 미리 저장해둔 그래프가 있으면, 요청 위치가 그 그래프 범위 안일 때만 재사용한다.
# 범위 밖이면 요청 위치 기준으로 OSM에서 새로 받아온다.
GRAPH_CACHE_FILE = os.getenv("GPS_ART_GRAPH_CACHE", "gps_art_graph.graphml")
_cached_graph = None
_cached_bbox = None  # (min_lat, max_lat, min_lng, max_lng)

if os.path.exists(GRAPH_CACHE_FILE):
    print(f"[캐시] {GRAPH_CACHE_FILE} 로드")
    _cached_graph = ox.load_graphml(GRAPH_CACHE_FILE)
    _ys = [d["y"] for _, d in _cached_graph.nodes(data=True)]
    _xs = [d["x"] for _, d in _cached_graph.nodes(data=True)]
    _cached_bbox = (min(_ys), max(_ys), min(_xs), max(_xs))
else:
    print(f"[캐시 없음] 요청마다 OSM에서 도로망을 받아옵니다.")

app = FastAPI(title="NaChimBal GPS Art Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 알고리즘 설정값 =====
CYCLE_LENGTH_MIN = 200      # 폐곡선 후보 최소 둘레(m)
CYCLE_LENGTH_MAX = 8000     # 폐곡선 후보 최대 둘레(m)
RATIO_WEIGHT = 20           # 변 길이 비율 DTW 가중치
MAX_VERTEX_RATIO = 2.5      # 목표 도형 대비 꼭짓점 개수 허용 배수 (초과분은 페널티)
VERTEX_PENALTY_WEIGHT = 2.0
SIMILARITY_THRESHOLD = 2.0  # 이 값보다 점수가 크면 생성 실패


# ===== 목표 도형 (0~1 정규화 좌표, 시작점으로 닫힌 형태) =====
def _star_shape(points=5, inner_ratio=0.45):
    coords = []
    for i in range(points * 2):
        r = 0.5 if i % 2 == 0 else 0.5 * inner_ratio
        theta = np.pi / 2 + i * np.pi / points
        coords.append([0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)])
    coords.append(coords[0])
    return np.array(coords)


SHAPES = {
    "heart": np.array([
        [0.5, 1.0], [0.2, 0.7], [0.0, 0.4], [0.2, 0.1],
        [0.5, 0.3], [0.8, 0.1], [1.0, 0.4], [0.8, 0.7], [0.5, 1.0],
    ]),
    "fish": np.array([
        [0.0, 0.5], [0.25, 0.8], [0.55, 0.8], [0.75, 0.55], [1.0, 0.85],
        [1.0, 0.15], [0.75, 0.45], [0.55, 0.2], [0.25, 0.2], [0.0, 0.5],
    ]),
    "star": _star_shape(),
    "diamond": np.array([
        [0.5, 1.0], [0.0, 0.5], [0.5, 0.0], [1.0, 0.5], [0.5, 1.0],
    ]),
    "triangle": np.array([
        [0.5, 1.0], [0.0, 0.0], [1.0, 0.0], [0.5, 1.0],
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
    return deduped


def _sq_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# ===== 형태 시그니처 & 유사도 =====
def shape_signature(coords):
    """회전각 시퀀스와 변 길이 비율 시퀀스 (위치·크기·회전에 무관한 도형 특징)"""
    coords = np.array(coords)
    if len(coords) < 4:
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
    """후보 폐곡선의 시작점·방향을 모두 돌려보며 목표 도형과의 최소 거리를 구함"""
    penalty = vertex_penalty(len(cand_angles), len(ref_angles))
    # 페널티만으로 임계값을 넘으면 어차피 채택될 수 없으므로 DTW 계산 생략
    if penalty > SIMILARITY_THRESHOLD:
        return float("inf")

    best = float("inf")
    for start in range(len(cand_angles)):
        for direction in (1, -1):
            rotated_angles = np.roll(cand_angles, -start)[::direction]
            rotated_ratios = np.roll(cand_ratios, -start)[::direction]
            best = min(best, signature_distance(rotated_angles, rotated_ratios, ref_angles, ref_ratios))

    return best + penalty


SHAPE_SIGNATURES = {name: shape_signature(coords) for name, coords in SHAPES.items()}


# ===== 도로망 그래프 =====
def build_deterministic_simple_graph(G_undirected):
    """
    노드/엣지를 정렬된 순서로 넣어, 그래프를 새로 받아오든 캐시에서 읽든
    cycle_basis() 결과가 항상 같게 나오도록 고정한다.
    """
    G_simple = nx.Graph()
    for n in sorted(G_undirected.nodes()):
        G_simple.add_node(n)
    for u, v in sorted(G_undirected.edges()):
        G_simple.add_edge(u, v)
    return G_simple


def is_inside_cache(latitude, longitude):
    if _cached_bbox is None:
        return False
    min_lat, max_lat, min_lng, max_lng = _cached_bbox
    return min_lat <= latitude <= max_lat and min_lng <= longitude <= max_lng


def load_graph(latitude, longitude, search_radius):
    if is_inside_cache(latitude, longitude):
        return _cached_graph
    return ox.graph_from_point(
        (latitude, longitude), dist=search_radius, network_type="walk", simplify=True
    )


# ===== GPS 아트 탐색 =====
def find_best_gps_art(latitude: float, longitude: float, search_radius: int, shape: str | None = None):
    """중심 좌표 주변에서 목표 도형과 가장 닮은 폐곡선을 찾는다. 없으면 None."""
    target_shapes = [shape] if shape else list(SHAPES.keys())

    G = load_graph(latitude, longitude, search_radius)
    G_undirected = ox.convert.to_undirected(G)
    G_simple = build_deterministic_simple_graph(G_undirected)

    candidates = []
    for cycle in nx.cycle_basis(G_simple):
        if len(cycle) < 4:
            continue
        length = cycle_length(cycle, G_undirected)
        if CYCLE_LENGTH_MIN <= length <= CYCLE_LENGTH_MAX:
            angles, ratios = shape_signature(cycle_to_coords(cycle, G_undirected))
            if angles is not None:
                candidates.append((cycle, length, angles, ratios))

    if not candidates:
        return None

    best = None  # (score, shape_name, cycle, length)
    for shape_name in target_shapes:
        ref_angles, ref_ratios = SHAPE_SIGNATURES[shape_name]
        for cycle, length, angles, ratios in candidates:
            score = best_match_score(angles, ratios, ref_angles, ref_ratios)
            if best is None or score < best[0]:
                best = (score, shape_name, cycle, length)

    best_score, best_shape, best_cycle, best_length = best
    if best_score > SIMILARITY_THRESHOLD:
        return None

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
        "distScore": round(best_score, 4),
    }


# ===== API =====
@app.post("/gps-art/generate", response_model=GpsArtResponse)
def generate_gps_art(req: GpsArtRequest):
    if req.shape is not None and req.shape not in SHAPES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 도형입니다: {req.shape} (지원: {', '.join(SHAPES)})",
        )

    result = find_best_gps_art(req.latitude, req.longitude, req.searchRadius, req.shape)
    if result is None:
        return GpsArtResponse(found=False)

    return GpsArtResponse(found=True, **result)


@app.get("/gps-art/shapes")
def list_shapes():
    return {"shapes": list(SHAPES.keys())}


@app.get("/health")
def health_check():
    return {"status": "ok"}