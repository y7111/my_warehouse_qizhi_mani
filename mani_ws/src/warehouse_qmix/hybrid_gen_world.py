#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_gen_world.py — 生成"混合四车"世界(全新文件,不改动任何现有文件)

把 4 台完整外观 wpb_mani(use_planar 轻量物理)直接嵌入 world，
规避多车 spawn 服务的 ODE 锁死(沿用项目既有"嵌world"思路，但全部输出到新文件)。
机械臂折叠姿势在 launch 启动后用 /gazebo/set_model_configuration 设置。

输出:
  warehouse_qmix/worlds/warehouse_hybrid.world
用法:
  cd ~/mani_ws && source devel/setup.bash
  python3 src/warehouse_qmix/hybrid_gen_world.py
"""
import os
import re
import subprocess

PKG_WPB  = subprocess.check_output(
    ['rospack', 'find', 'wpb_mani_description']).decode().strip()
XACRO    = os.path.join(PKG_WPB, 'urdf', 'wpb_mani.urdf.xacro')
BASE_WORLD = os.path.join(
    subprocess.check_output(['rospack', 'find', 'wpb_mani_simulator']).decode().strip(),
    'worlds', 'warehouse_cross.world')
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds')
OUT_WORLD = os.path.join(OUT_DIR, 'warehouse_hybrid.world')

# 4 台车：名字 / 起点 / 朝向(全部朝北)。全部 use_planar 轻量物理、外观一致。
ROBOTS = [
    ('robot_0', -1.5, -7.0, 1.5707963),
    ('robot_1', -0.5, -7.0, 1.5707963),
    ('robot_2',  0.5, -7.0, 1.5707963),
    ('robot_3',  1.5, -7.0, 1.5707963),
]


def make_sdf(ns, strip_control=True):
    urdf_path = '/tmp/hybrid_{}.urdf'.format(ns)
    urdf = subprocess.check_output(
        ['rosrun', 'xacro', 'xacro', XACRO,
         'robot_ns:={}'.format(ns), 'use_planar:=true'])
    with open(urdf_path, 'wb') as f:
        f.write(urdf)
    sdf = subprocess.check_output(['gz', 'sdf', '-p', urdf_path],
                                  stderr=subprocess.DEVNULL).decode()
    m = re.search(r'<model[\s\S]*</model>', sdf)
    if not m:
        raise RuntimeError('未在 SDF 中找到 <model> ({})'.format(ns))
    body = m.group(0)
    if strip_control:
        # 砍掉最吃 CPU 的 ros_control 及夹爪 mimic 插件(简化车不需要真控制,折叠靠 set_model_configuration)
        body = re.sub(r"<plugin[^>]*filename='libgazebo_ros_control.so'[\s\S]*?</plugin>", '', body)
        body = re.sub(r"<plugin[^>]*filename='libroboticsgroup_gazebo_mimic_joint_plugin.so'[\s\S]*?</plugin>", '', body)
    return body


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(BASE_WORLD) as f:
        world = f.read()

    blocks = []
    for name, x, y, yaw in ROBOTS:
        body = make_sdf(name)
        body = body.replace("name='wpb_mani'", "name='{}'".format(name), 1)
        # 用起点位姿替换 model 的第一个 <pose>
        body = re.sub(r'<pose>[^<]*</pose>',
                      '<pose>{} {} 0 0 0 {}</pose>'.format(x, y, yaw), body, count=1)
        blocks.append('    ' + body + '\n')
        print('  嵌入 {} @ ({}, {})'.format(name, x, y))

    world = world.replace('</world>', ''.join(blocks) + '</world>')
    with open(OUT_WORLD, 'w') as f:
        f.write(world)
    print('已生成: {}  ({} 行)'.format(OUT_WORLD, world.count(chr(10))))


if __name__ == '__main__':
    main()
