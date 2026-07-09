"""
Blender 无头渲染 AGV 机器人
用法: blender --background --python render_robot.py
"""
import bpy, math, os

MESH_DIR = "/home/jasper/mani_ws/src/wpb_mani/wpb_mani_description/meshes"
OUTPUT   = "/tmp/agv_robot_hd.png"

# ── 清空默认场景 ──────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

def import_dae(path, loc=(0,0,0), rot=(0,0,0)):
    bpy.ops.wm.collada_import(filepath=path, fix_orientation=True)
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            obj.location.x += loc[0]
            obj.location.y += loc[1]
            obj.location.z += loc[2]

def import_stl(path, loc=(0,0,0), scale=0.001):
    """STL 文件是毫米单位，需要 scale=0.001 转换为米"""
    bpy.ops.import_mesh.stl(filepath=path, global_scale=scale)
    for obj in bpy.context.selected_objects:
        obj.location.x += loc[0]
        obj.location.y += loc[1]
        obj.location.z += loc[2]

# ── 1. 底盘主体（含 Kinect 升降柱）──────────────
import_dae(os.path.join(MESH_DIR, "wpb_mani_base.dae"))

# ── 2. 4 个麦克纳姆轮（URDF 坐标）───────────────
# front_left:  x=0.2,  y=0.155,  z=0.029  mecanum_a
# front_right: x=0.2,  y=-0.155, z=0.029  mecanum_b
# back_left:   x=-0.2, y=0.155,  z=0.029  mecanum_b
# back_right:  x=-0.2, y=-0.155, z=0.029  mecanum_a
import_dae(os.path.join(MESH_DIR, "mecanum_a.dae"), loc=( 0.2,  0.155, 0.029))
import_dae(os.path.join(MESH_DIR, "mecanum_b.dae"), loc=( 0.2, -0.155, 0.029))
import_dae(os.path.join(MESH_DIR, "mecanum_b.dae"), loc=(-0.2,  0.155, 0.029))
import_dae(os.path.join(MESH_DIR, "mecanum_a.dae"), loc=(-0.2, -0.155, 0.029))

# ── 3. 机械臂 arm（STL 毫米→米, URDF 挂载点 x=0.203, z=0.1575）
arm_base = (0.203, 0.0, 0.1575)
for i in range(1, 6):
    import_stl(os.path.join(MESH_DIR, f"chain_link{i}.stl"),
               loc=arm_base, scale=0.001)
import_stl(os.path.join(MESH_DIR, "chain_link_grip_l.stl"), loc=arm_base, scale=0.001)
import_stl(os.path.join(MESH_DIR, "chain_link_grip_r.stl"), loc=arm_base, scale=0.001)

# ── 4. Azure Kinect 相机（URDF: x=-0.15, z≈0.585）──
import_dae(os.path.join(MESH_DIR, "azure_kinect.dae"), loc=(-0.15, 0, 0.585))

# ── 计算整体包围盒，自动设置相机 ─────────────────
all_x, all_y, all_z = [], [], []
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        for v in obj.data.vertices:
            co = obj.matrix_world @ v.co
            all_x.append(co.x); all_y.append(co.y); all_z.append(co.z)

cx = (min(all_x)+max(all_x))/2
cy = (min(all_y)+max(all_y))/2
cz = (min(all_z)+max(all_z))/2
max_size = max(max(all_x)-min(all_x), max(all_y)-min(all_y), max(all_z)-min(all_z))
print(f"场景包围盒: X={min(all_x):.3f}~{max(all_x):.3f}  Y={min(all_y):.3f}~{max(all_y):.3f}  Z={min(all_z):.3f}~{max(all_z):.3f}")
print(f"中心: ({cx:.3f}, {cy:.3f}, {cz:.3f})  最大尺寸: {max_size:.3f}m")

# ── 相机（自动对准中心，距离 = 最大尺寸 × 2.2）──────
cam_data = bpy.data.cameras.new("Camera")
cam_obj  = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

dist = max_size * 2.8
cam_obj.location = (cx + dist*0.55, cy - dist*0.55, cz + dist*0.40)
cam_data.lens = 35

constraint = cam_obj.constraints.new(type='TRACK_TO')
empty = bpy.data.objects.new("Target", None)
empty.location = (cx, cy, cz)
bpy.context.scene.collection.objects.link(empty)
constraint.target     = empty
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis    = 'UP_Y'

# ── 灯光 ─────────────────────────────────────────
sun = bpy.data.lights.new("Sun", type='SUN')
sun_obj = bpy.data.objects.new("Sun", sun)
bpy.context.scene.collection.objects.link(sun_obj)
sun_obj.rotation_euler = (math.radians(50), 0, math.radians(30))
sun.energy = 5.0

fill = bpy.data.lights.new("Fill", type='POINT')
fill_obj = bpy.data.objects.new("Fill", fill)
bpy.context.scene.collection.objects.link(fill_obj)
fill_obj.location = (-2, 1, 2)
fill.energy = 800

# ── 世界背景（深蓝渐变）─────────────────────────
world = bpy.context.scene.world
world.use_nodes = True
bg_node = world.node_tree.nodes['Background']
bg_node.inputs[0].default_value = (0.03, 0.05, 0.10, 1)
bg_node.inputs[1].default_value = 0.4

# ── 渲染（EEVEE + 高亮环境光）──────────────────
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.taa_render_samples = 128
scene.eevee.use_soft_shadows   = True
scene.eevee.use_ssr            = True    # 屏幕空间反射
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.filepath = OUTPUT
scene.render.image_settings.file_format = 'PNG'

print("开始渲染...")
bpy.ops.render.render(write_still=True)
print(f"渲染完成 → {OUTPUT}")
