import osmnx as ox

CENTER_POINT = (37.5796, 126.9770)  # 경복궁 중심
RADIUS = 1200
OUTPUT_FILE = "gyeongbokgung_graph.graphml"

print("도로 그래프 요청 중... (네트워크 상황에 따라 시간이 걸릴 수 있습니다)")
G = ox.graph_from_point(CENTER_POINT, dist=RADIUS, network_type="walk", simplify=True)

ox.save_graphml(G, OUTPUT_FILE)
print(f"완료: {OUTPUT_FILE} 로 저장됨 (노드 {len(G.nodes)}개, 엣지 {len(G.edges)}개)")