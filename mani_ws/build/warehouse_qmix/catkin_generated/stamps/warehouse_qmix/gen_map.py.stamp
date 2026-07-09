#!/usr/bin/env python3
"""
生成 warehouse_cross 导航地图（PGM + YAML）。

地图范围：x=[0,10]m，y=[-8,0]m，分辨率 0.05m/像素
对应 warehouse_cross.world 中的实际墙体。
运行：python3 gen_map.py
"""
import os
import numpy as np

W, H = 400, 320          # 像素：20m/0.05 × 16m/0.05（每格1.0m，分辨率0.05m/px）
RES  = 0.05
OX, OY = 0.0, -16.0      # 图像左下角世界坐标

img = np.full((H, W), 254, dtype=np.uint8)   # 全白=可通行

def fill(x1, x2, y1, y2):
    """将世界坐标矩形区域填充为障碍物（黑色）。"""
    c1 = max(0, int(round(x1 / RES)))
    c2 = min(W, int(round(x2 / RES)))
    # 图像行 0=顶部(世界y大)，行H-1=底部(世界y小)
    r1 = max(0, int(round(H - (y2 - OY) / RES)))
    r2 = min(H, int(round(H - (y1 - OY) / RES)))
    img[r1:r2, c1:c2] = 0

# ── 大块墙体（每格1.0m）─────────────────────────────────────────
fill( 0.0,  7.0,  -5.0,  0.0)   # NW：行0-4，列0-6
fill(13.0, 20.0,  -5.0,  0.0)   # NE：行0-4，列13-19
fill( 0.0,  7.0, -16.0, -10.0)  # SW：行10-15，列0-6
fill(13.0, 20.0, -16.0, -10.0)  # SE：行10-15，列13-19

# ── 瓶颈隔墙（R3/R11 中央，C9-C10）─────────────────────────────
fill( 9.0, 11.0,  -4.0,  -3.0)  # 北臂瓶颈
fill( 9.0, 11.0, -12.0, -11.0)  # 南臂瓶颈

# ── 走廊隔墙（R7，东西臂，C2-C4 / C15-C17）─────────────────────
fill( 2.0,  5.0,  -8.0,  -7.0)  # 西走廊隔墙
fill(15.0, 18.0,  -8.0,  -7.0)  # 东走廊隔墙

# ── 保存文件 ──────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'maps')
os.makedirs(out_dir, exist_ok=True)
pgm_path  = os.path.join(out_dir, 'warehouse_cross.pgm')
yaml_path = os.path.join(out_dir, 'warehouse_cross.yaml')

with open(pgm_path, 'wb') as f:
    header = f'P5\n{W} {H}\n255\n'.encode()
    f.write(header)
    f.write(img.tobytes())

with open(yaml_path, 'w') as f:
    f.write(f'image: {pgm_path}\n')
    f.write(f'resolution: {RES}\n')
    f.write(f'origin: [{OX}, {OY}, 0.0]\n')
    f.write('negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')

print(f'[gen_map] 地图已生成：{pgm_path}')
print(f'[gen_map] 配置已生成：{yaml_path}')
print(f'[gen_map] 尺寸：{W}×{H} px，分辨率：{RES}m/px，世界范围 20m×16m')
