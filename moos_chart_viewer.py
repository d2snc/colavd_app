#!/usr/bin/env python3
"""
Visualizador MOOS-GeoTIFF
─────────────────────────
• Lancha B (principal) → 127.0.0.1:9003
• Lancha A (obstáculo) → 127.0.0.1:9002
Mostra as duas embarcações sobre a carta 1511, com zoom centrado
na lancha B e triângulos girando pelo NAV_HEADING.
"""
from pathlib import Path
import threading

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import RegularPolygon
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

# ───────── aplicação ─────────
def main():
    chart, affine, crs = load_chart(CHART_FILE)
    nrows, ncols = chart.shape[:2]
    conv = LatLonConverter(crs)

    lst_B = MOOSListener(MAIN_MOOS_PORT)
    lst_A = MOOSListener(AUX_MOOS_PORT)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(chart)
    ax.set_title("USV – Carta 1511 (zoom dinâmico)")

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

    # Legenda
    ax.scatter([], [], marker=(3, 0, 0), s=300, color="red",
               label=f"Lancha B ({MAIN_MOOS_PORT})")
    ax.scatter([], [], marker=(3, 0, 0), s=300, color="#4fa7ff",
               label=f"Lancha A ({AUX_MOOS_PORT})")
    ax.legend(loc="upper right")

    # Função de animação
    def update(_):
        # ----- lancha B -----
        lat, lon, hdg = lst_B.latest()
        if lat is not None and lon is not None:
            r, c = conv.latlon_to_pix(lat, lon, affine)
            move_tri(tri_B, c, r)
            if hdg is not None:
                rotate_tri(tri_B, heading_to_triangle_angle(hdg))
            half = ZOOM_WINDOW_PX // 2
            ax.set_xlim(max(c - half, 0), min(c + half, ncols))
            ax.set_ylim(min(r + half, nrows), max(r - half, 0))  # Y invertido

        # ----- lancha A -----
        lat, lon, hdg = lst_A.latest()
        if lat is not None and lon is not None:
            r, c = conv.latlon_to_pix(lat, lon, affine)
            move_tri(tri_A, c, r)
            if hdg is not None:
                rotate_tri(tri_A, heading_to_triangle_angle(hdg))

        return tri_B, tri_A

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