#!/usr/bin/env python3
"""grid_map.py — 把 ROS 地图（.pgm/.yaml）切成 1m 网格，标出"哪些格子能走"。

被 traffic_manager 复用；也可单独运行自检：
    python3 grid_map.py            # 打印可行格子 ASCII 图
坐标约定（与 robot_agent 完全一致）：
    row = round(7.5 - y),  col = round(x + 9.5),  1 格 ≈ 1m
    格子 (r,c) 中心世界坐标： x = c - 9.5,  y = 7.5 - r
"""

import os

GRID_H, GRID_W = 16, 20

# 判定一个格子"能走"时，在格子中心周围采样的半宽（米）。
# 取 0.3m → 采样跨度 0.6m，约等于车直径，保证车摆在格心不蹭墙。
SAMPLE_HALF = 0.30
SAMPLE_STEPS = (-SAMPLE_HALF, 0.0, SAMPLE_HALF)

DEFAULT_YAML = os.path.join(os.path.dirname(__file__),
                            '..', 'maps', 'warehouse_cross.yaml')


def grid_to_xy(r, c):
    return (c - 9.5, 7.5 - r)


def xy_to_grid(x, y):
    r = int(round(7.5 - y))
    c = int(round(x + 9.5))
    return (max(0, min(GRID_H - 1, r)), max(0, min(GRID_W - 1, c)))


def _read_yaml(yaml_path):
    """极简解析：只取 image / resolution / origin / 阈值。"""
    info = {'free_thresh': 0.196, 'occupied_thresh': 0.65, 'negate': 0}
    base = os.path.dirname(os.path.abspath(yaml_path))
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('image:'):
                img = line.split(':', 1)[1].strip()
                info['image'] = img if os.path.isabs(img) else os.path.join(base, img)
            elif line.startswith('resolution:'):
                info['resolution'] = float(line.split(':', 1)[1])
            elif line.startswith('origin:'):
                nums = line.split('[', 1)[1].split(']', 1)[0].split(',')
                info['origin'] = [float(n) for n in nums]
            elif line.startswith('free_thresh:'):
                info['free_thresh'] = float(line.split(':', 1)[1])
            elif line.startswith('occupied_thresh:'):
                info['occupied_thresh'] = float(line.split(':', 1)[1])
            elif line.startswith('negate:'):
                info['negate'] = int(line.split(':', 1)[1])
    return info


def _read_pgm(path):
    """读 P5 二进制 pgm，返回 (width, height, maxval, bytes)。"""
    with open(path, 'rb') as f:
        data = f.read()
    # 解析头部 token（跳过注释行 #）
    idx = 0
    tokens = []
    while len(tokens) < 4:
        # 跳过空白
        while idx < len(data) and data[idx:idx+1].isspace():
            idx += 1
        if data[idx:idx+1] == b'#':
            while idx < len(data) and data[idx:idx+1] not in (b'\n', b'\r'):
                idx += 1
            continue
        start = idx
        while idx < len(data) and not data[idx:idx+1].isspace():
            idx += 1
        tokens.append(data[start:idx])
    magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    assert magic == b'P5', 'only binary P5 pgm supported'
    idx += 1  # 头部之后一个空白字节
    pix = data[idx:idx + w * h]
    return w, h, maxval, pix


def load_free_grid(yaml_path=DEFAULT_YAML):
    """返回 free[GRID_H][GRID_W]，True=该格能走。"""
    info = _read_yaml(yaml_path)
    w, h, maxval, pix = _read_pgm(info['image'])
    res = info['resolution']
    ox, oy = info['origin'][0], info['origin'][1]
    # ROS 占用约定（negate=0）：occ = (maxval - pixel)/maxval
    # free 阈值：occ < free_thresh  →  pixel > maxval*(1-free_thresh)
    free_pixel = maxval * (1.0 - info['free_thresh'])

    def pixel_free(x, y):
        # 世界 (x,y) → pgm 像素 (col_px, row_px)；pgm 行 0 在顶部(最大 y)
        cpx = int((x - ox) / res)
        rpx = int(h - 1 - (y - oy) / res)
        if cpx < 0 or cpx >= w or rpx < 0 or rpx >= h:
            return False
        return pix[rpx * w + cpx] > free_pixel

    free = [[False] * GRID_W for _ in range(GRID_H)]
    for r in range(GRID_H):
        for c in range(GRID_W):
            x, y = grid_to_xy(r, c)
            ok = True
            for dx in SAMPLE_STEPS:
                for dy in SAMPLE_STEPS:
                    if not pixel_free(x + dx, y + dy):
                        ok = False
                        break
                if not ok:
                    break
            free[r][c] = ok
    return free


def render(free, marks=None):
    """可行格子 ASCII 图。marks: dict (r,c)->字符，叠加显示。"""
    marks = marks or {}
    lines = []
    header = '    ' + ''.join('{:>2}'.format(c % 100) for c in range(GRID_W))
    lines.append(header)
    for r in range(GRID_H):
        row = ['{:>3} '.format(r)]
        for c in range(GRID_W):
            if (r, c) in marks:
                ch = marks[(r, c)]
            elif free[r][c]:
                ch = '.'
            else:
                ch = '#'
            row.append(' ' + ch)
        lines.append(''.join(row))
    return '\n'.join(lines)


if __name__ == '__main__':
    import sys
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_YAML
    free = load_free_grid(yaml_path)
    # 叠加货架停靠点/投递点，方便核对
    shelf_wp = [(6, 8), (6, 11),  # 占位，真实用 WP 坐标
                ]
    delivery = {(13, 7): 'A', (13, 9): 'B', (13, 11): 'C'}
    starts = {(15, 8): '0', (15, 9): '1', (15, 10): '2', (15, 11): '3'}
    marks = {}
    marks.update(delivery)
    marks.update(starts)
    print('可行格子图（. = 能走，# = 墙/不可走；A/B/C=投递点，0-3=车起点）')
    print('坐标：row=round(7.5-y), col=round(x+9.5)')
    print(render(free, marks))
    n_free = sum(sum(1 for c in row if c) for row in free)
    print('\n能走的格子数：%d / %d' % (n_free, GRID_H * GRID_W))
