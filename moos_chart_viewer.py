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
import pymoos
from pyproj import CRS, Transformer

# ───────── CONFIGURAÇÃO ───────── #
CHART_FILE      = Path("data/1511geotiff.tif")
MAIN_MOOS_PORT  = 9003          # lancha B
AUX_MOOS_PORT   = 9002          # lancha A
VAR_LAT, VAR_LONG, VAR_HEAD = "NAV_LAT", "NAV_LONG", "NAV_HEADING"

ZOOM_WINDOW_PX  = 1000          # lado da janela de zoom (px)
MAIN_SIZE_PX    = 40            # triângulo lancha B
AUX_SIZE_PX     = 34            # triângulo lancha A
UPDATE_MS       = 400           # período da animação (ms)

# Collision Avoidance
COLLISION_DISTANCE_M = 500      # distância para ativar desvio (metros)
SAFETY_MARGIN_M     = 100       # margem de segurança adicional
GRID_SIZE_M         = 50        # tamanho da célula do grid A* (metros)
LOOKAHEAD_DISTANCE_M = 1000     # distância à frente para calcular rota

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

# ───── A* Path Planning ─────
class AStarPlanner:
    def __init__(self, grid_size_m: float = GRID_SIZE_M):
        self.grid_size = grid_size_m
        
    def plan_avoidance_path(self, 
                           start_lat: float, start_lon: float, start_heading: float,
                           obstacle_lat: float, obstacle_lon: float,
                           lookahead_distance: float = LOOKAHEAD_DISTANCE_M) -> List[Tuple[float, float]]:
        """
        Planeja rota de desvio usando A*
        Retorna lista de waypoints (lat, lon)
        """
        # Ponto de destino à frente da lancha
        target_lat, target_lon = point_at_distance_bearing(
            start_lat, start_lon, lookahead_distance, start_heading
        )
        
        # Criar grid local ao redor da área
        min_lat = min(start_lat, target_lat, obstacle_lat) - 0.01
        max_lat = max(start_lat, target_lat, obstacle_lat) + 0.01
        min_lon = min(start_lon, target_lon, obstacle_lon) - 0.01
        max_lon = max(start_lon, target_lon, obstacle_lon) + 0.01
        
        # Converter para grid
        start_grid = self._latlon_to_grid(start_lat, start_lon, min_lat, min_lon)
        target_grid = self._latlon_to_grid(target_lat, target_lon, min_lat, min_lon)
        obstacle_grid = self._latlon_to_grid(obstacle_lat, obstacle_lon, min_lat, min_lon)
        
        # Executar A*
        path_grid = self._astar(start_grid, target_grid, obstacle_grid)
        
        if not path_grid:
            # Se A* falhar, criar rota simples de desvio
            return self._simple_avoidance_path(start_lat, start_lon, start_heading, 
                                             obstacle_lat, obstacle_lon, target_lat, target_lon)
        
        # Converter path de volta para lat/lon
        path_latlon = []
        for grid_point in path_grid[1:]:  # Pular o ponto inicial
            lat, lon = self._grid_to_latlon(grid_point[0], grid_point[1], min_lat, min_lon)
            path_latlon.append((lat, lon))
        
        return path_latlon
    
    def _latlon_to_grid(self, lat: float, lon: float, min_lat: float, min_lon: float) -> Tuple[int, int]:
        """Converte lat/lon para coordenadas de grid"""
        # Aproximação simples - para uso local
        lat_dist = haversine_distance(min_lat, min_lon, lat, min_lon)
        lon_dist = haversine_distance(min_lat, min_lon, min_lat, lon)
        
        grid_x = int(lon_dist / self.grid_size)
        grid_y = int(lat_dist / self.grid_size)
        return (grid_x, grid_y)
    
    def _grid_to_latlon(self, grid_x: int, grid_y: int, min_lat: float, min_lon: float) -> Tuple[float, float]:
        """Converte coordenadas de grid para lat/lon"""
        lat_dist = grid_y * self.grid_size
        lon_dist = grid_x * self.grid_size
        
        # Aproximação simples para conversão de volta
        lat_per_meter = 1.0 / 111000  # Aproximadamente 1 grau = 111km
        lon_per_meter = lat_per_meter / math.cos(math.radians(min_lat))
        
        lat = min_lat + lat_dist * lat_per_meter
        lon = min_lon + lon_dist * lon_per_meter
        return (lat, lon)
    
    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int], 
               obstacle: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Algoritmo A* simplificado"""
        heap = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}
        
        # Criar área de obstáculo (círculo ao redor do obstáculo)
        obstacle_radius = int((COLLISION_DISTANCE_M + SAFETY_MARGIN_M) / self.grid_size)
        obstacle_cells = set()
        for dx in range(-obstacle_radius, obstacle_radius + 1):
            for dy in range(-obstacle_radius, obstacle_radius + 1):
                if dx*dx + dy*dy <= obstacle_radius*obstacle_radius:
                    obstacle_cells.add((obstacle[0] + dx, obstacle[1] + dy))
        
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        while heap:
            current_cost, current = heapq.heappop(heap)
            
            if current == goal:
                # Reconstruir caminho
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Verificar se está em obstáculo
                if neighbor in obstacle_cells:
                    continue
                
                new_cost = cost_so_far[current] + (1.4 if abs(dx) + abs(dy) == 2 else 1)
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal)
                    heapq.heappush(heap, (priority, neighbor))
                    came_from[neighbor] = current
        
        return []  # Caminho não encontrado
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Distância Manhattan como heurística"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _simple_avoidance_path(self, start_lat: float, start_lon: float, start_heading: float,
                              obstacle_lat: float, obstacle_lon: float,
                              target_lat: float, target_lon: float) -> List[Tuple[float, float]]:
        """Rota de desvio simples quando A* falha"""
        # Determinar lado para desvio (vira à direita por padrão)
        avoidance_heading = (start_heading + 90) % 360
        
        # Ponto de desvio
        avoidance_distance = COLLISION_DISTANCE_M + SAFETY_MARGIN_M
        waypoint1_lat, waypoint1_lon = point_at_distance_bearing(
            start_lat, start_lon, avoidance_distance, avoidance_heading
        )
        
        # Ponto para retornar ao curso
        waypoint2_lat, waypoint2_lon = point_at_distance_bearing(
            waypoint1_lat, waypoint1_lon, avoidance_distance, start_heading
        )
        
        return [(waypoint1_lat, waypoint1_lon), (waypoint2_lat, waypoint2_lon), (target_lat, target_lon)]

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
            
            # Verificar se precisa ativar collision avoidance
            if distance_to_A < COLLISION_DISTANCE_M and hdg_B is not None:
                if not collision_active:
                    collision_active = True
                    info_lines.append("🚨 COLLISION AVOIDANCE ATIVO")
                    
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
                    info_lines.append("🚨 COLLISION AVOIDANCE ATIVO")
            else:
                if collision_active:
                    collision_active = False
                    last_waypoints = []
                    avoidance_line.set_data([], [])  # Limpar rota
                    info_lines.append("✅ Rota Normal")
                else:
                    info_lines.append("✅ Navegação Normal")
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