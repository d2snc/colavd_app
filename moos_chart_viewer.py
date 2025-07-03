#!/usr/bin/env python3
"""
Visualizador MOOS-GeoTIFF com Collision Avoidance
─────────────────────────────────────────────────
• Lancha B (principal) → 127.0.0.1:9003
• Lancha A (obstáculo) → 127.0.0.1:9002
Mostra as duas embarcações sobre a carta 1511, com zoom centrado
na lancha B e triângulos girando pelo NAV_HEADING.
Implementa Collision Avoidance com A* quando distância < 500m.
"""
from pathlib import Path
import threading
import math
from typing import List, Tuple, Optional
from collections import deque
import heapq

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import RegularPolygon, Circle
from matplotlib.lines import Line2D
import pymoos  # Commented out for testing
from pyproj import CRS, Transformer

# ───────── CONFIGURAÇÃO ───────── #
CHART_FILE      = Path("data/1511geotiff.tif")
MAIN_MOOS_PORT  = 9003          # lancha B
AUX_MOOS_PORT   = 9002          # lancha A
VAR_LAT, VAR_LONG, VAR_HEAD = "NAV_LAT", "NAV_LONG", "NAV_HEADING"

ZOOM_WINDOW_PX  = 1000          # lado da janela de zoom (px)
MAIN_SIZE_PX    = 20            # triângulo lancha B
AUX_SIZE_PX     = 20            # triângulo lancha A
UPDATE_MS       = 400           # período da animação (ms)

# Collision Avoidance
COLLISION_DISTANCE_M = 100      # distância para ativar desvio (metros)
COLLISION_CLEAR_M   = 150       # distância para desativar desvio (histerese)
SAFETY_MARGIN_M     = 100       # margem de segurança adicional
GRID_SIZE_M         = 1        # tamanho da célula do grid A* (metros)
LOOKAHEAD_DISTANCE_M = 200     # distância à frente para calcular rota

# Origem do sistema de coordenadas local (Rio de Janeiro)
LAT_ORIGIN = -22.93335
LON_ORIGIN = -43.136666665
# ───────────────────────────────── #

# ────── utilidades de carta ──────
def load_chart(path: Path):
    src  = rasterio.open(path)
    img  = src.read(out_dtype="uint8").transpose(1, 2, 0)
    return img, src.transform, CRS.from_wkt(src.crs.wkt)

class LatLonConverter:
    def __init__(self, crs: CRS):
        self._tf = (Transformer.from_crs(4326, crs, always_xy=True)
                    if crs.to_epsg() != 4326 else None)
    
    def latlon_to_pix(self, lat, lon, affine):
        x, y = (self._tf.transform(lon, lat) if self._tf else (lon, lat))
        col, row = ~affine * (x, y)
        return float(row), float(col)
    
    def latlon_to_xy(self, lat, lon):
        """
        Converte lat/lon para coordenadas locais x,y (metros) usando origem do Rio de Janeiro
        
        Sistema local:
        - Origem: Lat=-22.93335, Lon=-43.136666665 (Rio de Janeiro)
        - X positivo = Leste
        - Y positivo = Norte
        """
        # Calcular diferenças em relação à origem
        dlat = lat - LAT_ORIGIN
        dlon = lon - LON_ORIGIN
        
        # Converter para metros usando aproximação local
        # 1 grau de latitude ≈ 111,000 metros
        # 1 grau de longitude = 111,000 * cos(latitude) metros
        lat_rad = math.radians(LAT_ORIGIN)
        
        y = dlat * 111000.0  # Norte positivo
        x = dlon * 111000.0 * math.cos(lat_rad)  # Leste positivo
        
        return float(x), float(y)
    
    def xy_to_latlon(self, x, y):
        """
        Converte coordenadas locais x,y (metros) para lat/lon
        """
        lat_rad = math.radians(LAT_ORIGIN)
        
        dlat = y / 111000.0
        dlon = x / (111000.0 * math.cos(lat_rad))
        
        lat = LAT_ORIGIN + dlat
        lon = LON_ORIGIN + dlon
        
        return float(lat), float(lon)

# ───── listener MOOS ─────
class MOOSListener(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.lat = self.lon = self.hdg = None
        self._lock = threading.Lock()
        self.comms = pymoos.comms()
        self.comms.set_on_connect_callback(self._on_connect)
        self.comms.set_on_mail_callback(self._on_mail)
        self.comms.run("127.0.0.1", port, f"ChartViewer-{port}")
    
    def _on_connect(self):
        for v in (VAR_LAT, VAR_LONG, VAR_HEAD):
            self.comms.register(v, 0)
        return True
    
    def _on_mail(self):
        with self._lock:
            for m in self.comms.fetch():
                if m.key() == VAR_LAT:
                    self.lat = m.double()
                elif m.key() == VAR_LONG:
                    self.lon = m.double()
                elif m.key() == VAR_HEAD:
                    self.hdg = m.double() % 360
        return True
    
    def latest(self):
        with self._lock:
            return self.lat, self.lon, self.hdg
    
    def send_waypoints(self, waypoints: List[Tuple[float, float]], converter):
        """Envia waypoints como WPT_UPDATE para MOOS-IvP em coordenadas x,y locais"""
        if not waypoints:
            return
        
        # Converter lat/lon para coordenadas locais x,y
        xy_waypoints = []
        for lat, lon in waypoints:
            x, y = converter.latlon_to_xy(lat, lon)
            xy_waypoints.append((x, y))
        
        # Formato: WPT_UPDATE=polygon=x1,y1:x2,y2:x3,y3
        polygon_str = ":".join([f"{x:.2f},{y:.2f}" for x, y in xy_waypoints])
        wpt_command = f"polygon={polygon_str}"
        
        print(f"Enviando WPT_UPDATE: {wpt_command}")
        print(f"Waypoints (x,y): {xy_waypoints}")
        self.comms.notify("WPT_UPDATE", wpt_command, -1)
        return True

# ─── helpers de compatibilidade ───
def move_tri(tri: RegularPolygon, col, row):
    if hasattr(tri, "set_xy"):
        tri.set_xy((col, row))
    else:
        tri.xy = (col, row)
        if hasattr(tri, "_recompute_path"):
            tri._recompute_path()

def rotate_tri(tri: RegularPolygon, theta):
    if hasattr(tri, "set_theta"):           # Matplotlib ≥ 3.8
        tri.set_theta(theta)
    elif hasattr(tri, "set_orientation"):   # 3.4 – 3.7
        tri.set_orientation(theta)
    else:                                   # mais antigas
        tri.orientation = theta
        if hasattr(tri, "_recompute_path"):
            tri._recompute_path()

# ───── conversão de heading ─────
def heading_to_triangle_angle(heading):
    """
    Converte heading náutico para ângulo do triângulo no matplotlib.
    
    Sistema náutico: 0° = Norte, 90° = Leste, 180° = Sul, 270° = Oeste
    No matplotlib com RegularPolygon, o heading direto funciona corretamente
    
    Mapeamento direto:
    - Heading 0° (Norte) → ângulo 0° → triângulo aponta para cima
    - Heading 90° (Leste) → ângulo 90° → triângulo aponta para direita
    - Heading 180° (Sul) → ângulo 180° → triângulo aponta para baixo
    - Heading 270° (Oeste) → ângulo 270° → triângulo aponta para esquerda
    """
    return np.deg2rad(heading+180)

# ───── utilitários geográficos ─────
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distância em metros entre dois pontos lat/lon usando fórmula de Haversine"""
    R = 6371000  # Raio da Terra em metros
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def bearing_between_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula bearing (heading) de um ponto para outro"""
    dlon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
    
    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360

def point_at_distance_bearing(lat: float, lon: float, distance_m: float, bearing_deg: float) -> Tuple[float, float]:
    """Calcula novo ponto a uma distância e bearing de um ponto inicial"""
    R = 6371000  # Raio da Terra em metros
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)
    
    new_lat = math.asin(math.sin(lat_rad) * math.cos(distance_m/R) +
                       math.cos(lat_rad) * math.sin(distance_m/R) * math.cos(bearing_rad))
    
    new_lon = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_m/R) * math.cos(lat_rad),
                                  math.cos(distance_m/R) - math.sin(lat_rad) * math.sin(new_lat))
    
    return math.degrees(new_lat), math.degrees(new_lon)

# ───────────────────────────── AStarPlanner ─────────────────────────────
class AStarPlanner:
    """
    CAA-A* (Han et al., 2020) com suavização de trajetória
    ------------------------------------------------------
    • Changeable Action Space: cone ±Δψ, camadas e splits variam
      em função do risco de colisão fuzzy (TCPA/DCPA – Tabela II).
    • Custo: g_geom + w_risk·CR  (Lee & Rhee 2001).
    • Pós-processamento:  Douglas-Peucker  →  Catmull-Rom spline
      para um caminho suave e operacional.
    """

    # ——— parâmetros extraídos / ajustados dos artigos ———
    _tcp_sets = [0, 20, 60, 120]              # limites TCPA (s)
    _dcp_sets = [0, 0.1, 0.3, 0.6]            # limites DCPA (nm ≈ 111 m)
    _risk_table = np.array(                   # grau fuzzy CR ∈ [0,1]
        [[1.0, .95, .85, .70, .55],
         [1.0, .90, .75, .60, .45],
         [1.0, .85, .60, .45, .30],
         [.95, .75, .50, .35, .20],
         [.90, .60, .40, .25, .10]]
    )

    def __init__(self,
                 grid_size_m: float = GRID_SIZE_M,
                 w_risk: float = 50.0):
        self.grid_size = grid_size_m
        self.w_risk = w_risk

    # ───────────────── API pública ─────────────────
    def plan_avoidance_path(self,
                            start_lat, start_lon, start_heading,
                            obs_lat,   obs_lon,
                            lookahead_distance=LOOKAHEAD_DISTANCE_M):

        # destino “look-ahead” à frente da embarcação
        tgt_lat, tgt_lon = point_at_distance_bearing(
            start_lat, start_lon, lookahead_distance, start_heading)

        # caixa de busca (±~1.6 km)
        pad = 0.015
        min_lat = min(start_lat, tgt_lat, obs_lat) - pad
        max_lat = max(start_lat, tgt_lat, obs_lat) + pad
        min_lon = min(start_lon, tgt_lon, obs_lon) - pad
        max_lon = max(start_lon, tgt_lon, obs_lon) + pad

        # converte para grade inteira
        s = self._latlon_to_grid(start_lat, start_lon, min_lat, min_lon)
        g = self._latlon_to_grid(tgt_lat,   tgt_lon,   min_lat, min_lon)
        o = self._latlon_to_grid(obs_lat,   obs_lon,   min_lat, min_lon)

        # executa A*
        path = self._astar_caa(s, g, o, obs_lat, obs_lon,
                               min_lat, min_lon, max_lat, max_lon)

        if not path:   # fallback
            return self._simple_avoidance_path(
                start_lat, start_lon, start_heading,
                obs_lat, obs_lon, tgt_lat, tgt_lon)

        # grade → lat/lon
        path_latlon = [self._grid_to_latlon(x, y, min_lat, min_lon)
                       for x, y in path[1:]]           # pula nó inicial

        # suaviza (RDP + Catmull-Rom)
        return self._smooth_path(path_latlon)

    # ────────── núcleo CAA-A* ──────────
    def _astar_caa(self, s, g, o, obs_lat, obs_lon,
                   min_lat, min_lon, max_lat, max_lon):

        open_heap = [(0.0, s)]
        came, g_cost = {s: None}, {s: 0.0}

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == g:
                return self._reconstruct(came, cur)

            # risco fuzzy neste nó
            cur_lat, cur_lon = self._grid_to_latlon(
                cur[0], cur[1], min_lat, min_lon)
            risk = self._collision_risk(cur_lat, cur_lon, obs_lat, obs_lon)

            # define Δψ, profundidade e splits (Fig. 2)
            dpsi   = np.interp(risk, [0,1], [15, 90])      # graus
            layers = int(np.interp(risk, [0,1], [2, 6]))
            splits = int(np.interp(risk, [0,1], [3, 9]))

            # gera vizinhos
            for layer in range(1, layers+1):
                step = layer
                for k in range(-splits//2, splits//2+1):
                    ang = math.radians(k * dpsi / splits)
                    dx  = round(step*math.cos(ang))
                    dy  = round(step*math.sin(ang))
                    nxt = (cur[0]+dx, cur[1]+dy)

                    # dentro da zona segura do obstáculo?
                    if (nxt[0]-o[0])**2 + (nxt[1]-o[1])**2 <= \
                       ((COLLISION_DISTANCE_M+SAFETY_MARGIN_M)/self.grid_size)**2:
                        continue

                    tentative_g = (g_cost[cur] + self.grid_size*layer +
                                   self.w_risk * risk)

                    if nxt not in g_cost or tentative_g < g_cost[nxt]:
                        g_cost[nxt] = tentative_g
                        came[nxt]   = cur
                        f = tentative_g + self._heuristic(nxt, g)
                        heapq.heappush(open_heap, (f, nxt))
        return []  # falhou

    # ────────── risco fuzzy TCPA/DCPA ──────────
    def _collision_risk(self, lat, lon, obs_lat, obs_lon):
        tcp = max(self._time_to_cpa(lat, lon, obs_lat, obs_lon), 0.1)
        dcp = max(haversine_distance(lat, lon, obs_lat, obs_lon)/1852, 0.01)

        tcp_i = min(np.digitize(tcp, self._tcp_sets), 4)
        dcp_i = min(np.digitize(dcp, self._dcp_sets), 4)
        return float(self._risk_table[dcp_i, tcp_i])

    def _time_to_cpa(self, lat1, lon1, lat2, lon2, v_rel=2.0):
        """Estimativa grosseira de TCPA (s)."""
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        return dist / max(v_rel, 0.1)

    # ────────── heurística & grade ──────────
    def _heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])   # Manhattan

    def _latlon_to_grid(self, lat, lon, min_lat, min_lon):
        lat_m = haversine_distance(min_lat, min_lon, lat, min_lon)
        lon_m = haversine_distance(min_lat, min_lon, min_lat, lon)
        return int(lon_m/self.grid_size), int(lat_m/self.grid_size)

    def _grid_to_latlon(self, gx, gy, min_lat, min_lon):
        lat = min_lat + (gy*self.grid_size)/111_000
        lon = min_lon + (gx*self.grid_size)/(111_000*math.cos(math.radians(min_lat)))
        return lat, lon

    def _reconstruct(self, came, node):
        path = []
        while node:
            path.append(node)
            node = came[node]
        return path[::-1]

    # ────────── suavização de caminho ──────────
    def _simplify_rdp(self, pts, eps=2.0):
        """Douglas-Peucker em XY locais (metros)."""
        if len(pts) <= 2:
            return pts
        lat0 = pts[0][0]
        kx = 111_000*math.cos(math.radians(lat0)); ky = 111_000
        xy = np.c_[(np.array([p[1] for p in pts])-pts[0][1])*kx,
                   (np.array([p[0] for p in pts])-pts[0][0])*ky]

        # RDP recursivo
        def rdp(lo, hi):
            v = xy[hi]-xy[lo]; n = np.linalg.norm(v) or 1
            d = np.abs(np.cross(v, xy[lo+1:hi]-xy[lo])/n)
            if d.max() < eps:
                return [lo, hi]
            idx = lo+1+d.argmax()
            return rdp(lo, idx)[:-1]+rdp(idx, hi)

        keep = rdp(0, len(pts)-1)
        return [pts[i] for i in keep]

        # ────────── nova versão: densificação “segment wise” ──────────
    def _smooth_path(self, pts, res_m=5.0):
        """
        1) simplifica com RDP (mesmo código de antes, eps=2*grid)
        2) percorre cada segmento p0-p1 e insere pontos a cada res_m
        Retorna lista de (lat, lon) com espaçamento ~res_m.
        """
        if len(pts) < 3:
            return pts

        # 1) simplificação
        simp = self._simplify_rdp(pts, eps=2*self.grid_size)

        # 2) densificação linear
        out = [simp[0]]
        for p0, p1 in zip(simp[:-1], simp[1:]):
            dist = haversine_distance(*p0, *p1)
            if dist < 1e-2:          # pontos coincidentes
                continue
            n_seg = max(int(dist // res_m), 1)
            for k in range(1, n_seg+1):
                frac = k / n_seg
                lat = p0[0] + frac * (p1[0] - p0[0])
                lon = p0[1] + frac * (p1[1] - p0[1])
                out.append((lat, lon))
        return out


    # ────────── fallback simples ──────────
    def _simple_avoidance_path(self, start_lat, start_lon, start_heading,
                               obstacle_lat, obstacle_lon,
                               target_lat, target_lon):
        avoidance_heading   = (start_heading + 90) % 360
        avoidance_distance  = COLLISION_DISTANCE_M + SAFETY_MARGIN_M
        wp1 = point_at_distance_bearing(start_lat, start_lon,
                                        avoidance_distance, avoidance_heading)
        wp2 = point_at_distance_bearing(*wp1,
                                        avoidance_distance, start_heading)
        return [wp1, wp2, (target_lat, target_lon)]



# ───────── aplicação ─────────
def main():
    chart, affine, crs = load_chart(CHART_FILE)
    nrows, ncols = chart.shape[:2]
    conv = LatLonConverter(crs)
    planner = AStarPlanner()

    lst_B = MOOSListener(MAIN_MOOS_PORT)
    lst_A = MOOSListener(AUX_MOOS_PORT)

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(chart)
    ax.set_title("USV – Carta 1511 (zoom dinâmico) | Collision Avoidance Ativo")

    # Triângulos — apenas 3 argumentos posicionais!
    tri_B = RegularPolygon(
        xy=(0, 0),
        numVertices=3,
        radius=MAIN_SIZE_PX,
        facecolor="red",
        edgecolor="black",
        zorder=4,
    )
    tri_A = RegularPolygon(
        xy=(0, 0),
        numVertices=3,
        radius=AUX_SIZE_PX,
        facecolor="#4fa7ff",
        edgecolor="black",
        zorder=4,
    )
    ax.add_patch(tri_B)
    ax.add_patch(tri_A)

    # Círculo de detecção de colisão
    collision_circle = Circle((0, 0), radius=0, fill=False, 
                             edgecolor='orange', linewidth=2, 
                             linestyle='--', alpha=0.7, zorder=2)
    ax.add_patch(collision_circle)

    # Linha da rota de desvio
    avoidance_line, = ax.plot([], [], 'g-', linewidth=3, 
                             alpha=0.8, zorder=3, label='Rota de Desvio')

    # Texto de informações
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legenda
    ax.scatter([], [], marker=(3, 0, 0), s=300, color="red",
               label=f"Lancha B ({MAIN_MOOS_PORT})")
    ax.scatter([], [], marker=(3, 0, 0), s=300, color="#4fa7ff",
               label=f"Lancha A ({AUX_MOOS_PORT})")
    ax.plot([], [], '--', color='orange', label=f'Zona de Colisão ({COLLISION_DISTANCE_M}m)')
    ax.legend(loc="upper right")

    # Estado do collision avoidance
    collision_active = False
    last_waypoints = []

    # Função de animação
    def update(_):
        nonlocal collision_active, last_waypoints
        
        # ----- lancha B -----
        lat_B, lon_B, hdg_B = lst_B.latest()
        # ----- lancha A -----
        lat_A, lon_A, hdg_A = lst_A.latest()
        
        distance_to_A = None
        info_lines = []
        
        if lat_B is not None and lon_B is not None:
            r_B, c_B = conv.latlon_to_pix(lat_B, lon_B, affine)
            move_tri(tri_B, c_B, r_B)
            if hdg_B is not None:
                rotate_tri(tri_B, heading_to_triangle_angle(hdg_B))
            
            # Configurar zoom centrado na lancha B
            half = ZOOM_WINDOW_PX // 2
            ax.set_xlim(max(c_B - half, 0), min(c_B + half, ncols))
            ax.set_ylim(min(r_B + half, nrows), max(r_B - half, 0))  # Y invertido
            
            info_lines.append(f"Lancha B: {lat_B:.6f}, {lon_B:.6f}")
            if hdg_B is not None:
                info_lines.append(f"Heading B: {hdg_B:.1f}°")
            
            # Mostrar coordenadas locais também
            x_B, y_B = conv.latlon_to_xy(lat_B, lon_B)
            info_lines.append(f"Local B: x={x_B:.1f}m, y={y_B:.1f}m")

        if lat_A is not None and lon_A is not None:
            r_A, c_A = conv.latlon_to_pix(lat_A, lon_A, affine)
            move_tri(tri_A, c_A, r_A)
            if hdg_A is not None:
                rotate_tri(tri_A, heading_to_triangle_angle(hdg_A))
            
            info_lines.append(f"Lancha A: {lat_A:.6f}, {lon_A:.6f}")
            if hdg_A is not None:
                info_lines.append(f"Heading A: {hdg_A:.1f}°")
            
            # Mostrar coordenadas locais também
            x_A, y_A = conv.latlon_to_xy(lat_A, lon_A)
            info_lines.append(f"Local A: x={x_A:.1f}m, y={y_A:.1f}m")

        # Calcular distância e verificar colisão
        if lat_B is not None and lon_B is not None and lat_A is not None and lon_A is not None:
            distance_to_A = haversine_distance(lat_B, lon_B, lat_A, lon_A)
            info_lines.append(f"Distância: {distance_to_A:.1f}m")
            
            # Mostrar círculo de detecção ao redor da lancha A
            # Converter raio de metros para pixels (aproximação)
            radius_deg = COLLISION_DISTANCE_M / 111000  # Aproximação: 1° ≈ 111km
            r_circle, c_circle = conv.latlon_to_pix(lat_A + radius_deg, lon_A, affine)
            radius_px = abs(r_circle - r_A)
            
            collision_circle.set_center((c_A, r_A))
            collision_circle.set_radius(radius_px)
            
            # Verificar se precisa ativar collision avoidance (com histerese)
            activation_distance = COLLISION_DISTANCE_M if not collision_active else COLLISION_CLEAR_M
            
            if distance_to_A < activation_distance and hdg_B is not None:
                if not collision_active:
                    collision_active = True
                    info_lines.append("*** COLLISION AVOIDANCE ATIVO ***")
                    
                    # Calcular rota de desvio
                    waypoints = planner.plan_avoidance_path(
                        lat_B, lon_B, hdg_B, lat_A, lon_A
                    )
                    
                    if waypoints:
                        last_waypoints = waypoints
                        # Enviar waypoints para MOOS (em coordenadas x,y)
                        lst_B.send_waypoints(waypoints, conv)
                        
                        # Plotar rota de desvio
                        route_cols, route_rows = [], []
                        route_rows.append(r_B)
                        route_cols.append(c_B)
                        
                        for wp_lat, wp_lon in waypoints:
                            wp_r, wp_c = conv.latlon_to_pix(wp_lat, wp_lon, affine)
                            route_rows.append(wp_r)
                            route_cols.append(wp_c)
                        
                        avoidance_line.set_data(route_cols, route_rows)
                else:
                    info_lines.append("*** COLLISION AVOIDANCE ATIVO ***")
                    
                    # Manter a rota plotada enquanto estiver ativo
                    if last_waypoints:
                        route_cols, route_rows = [], []
                        route_rows.append(r_B)
                        route_cols.append(c_B)
                        
                        for wp_lat, wp_lon in last_waypoints:
                            wp_r, wp_c = conv.latlon_to_pix(wp_lat, wp_lon, affine)
                            route_rows.append(wp_r)
                            route_cols.append(wp_c)
                        
                        avoidance_line.set_data(route_cols, route_rows)
            else:
                if collision_active:
                    collision_active = False
                    last_waypoints = []
                    avoidance_line.set_data([], [])  # Limpar rota
                    info_lines.append("--- Retornando rota normal ---")
                else:
                    info_lines.append("Navegacao Normal")
        else:
            # Limpar círculo se não há dados
            collision_circle.set_radius(0)
            avoidance_line.set_data([], [])
        
        # Atualizar texto de informações
        info_text.set_text("\n".join(info_lines))

        return tri_B, tri_A, collision_circle, avoidance_line, info_text

    # Mantém referência de animação
    anim = FuncAnimation(
        fig, update,
        interval=UPDATE_MS,
        blit=False,
        cache_frame_data=False
    )
    plt.show()

if __name__ == "__main__":
    main()
