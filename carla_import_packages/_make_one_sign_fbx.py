
import bpy
import sys
from pathlib import Path

argv = sys.argv
argv = argv[argv.index("--") + 1:]

sign_name = argv[0]
texture_path = Path(argv[1]).resolve()
fbx_path = Path(argv[2]).resolve()

# Sahneyi temizle
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Ölçüler metre kabul edilir.
# Tabela: 0.95 x 0.95 m, ince panel.
# Direk: 1.65 m.
board_w = 0.95
board_h = 0.95
board_t = 0.035
pole_h = 1.65
pole_r = 0.035

# Direk
bpy.ops.mesh.primitive_cylinder_add(
    vertices=24,
    radius=pole_r,
    depth=pole_h,
    location=(0.0, 0.0, pole_h / 2.0),
)
pole = bpy.context.object
pole.name = f"{sign_name}_pole"

mat_pole = bpy.data.materials.new(f"M_{sign_name}_pole")
mat_pole.diffuse_color = (0.12, 0.12, 0.12, 1.0)
pole.data.materials.append(mat_pole)

# Tabela paneli
# Panel merkezini direğin üstüne yakın koyuyoruz.
board_center_z = pole_h + board_h * 0.45

bpy.ops.mesh.primitive_cube_add(
    size=1.0,
    location=(0.0, -0.015, board_center_z),
)
board = bpy.context.object
board.name = f"{sign_name}_board"
board.dimensions = (board_w, board_t, board_h)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Texture material
mat = bpy.data.materials.new(f"M_{sign_name}_texture")
mat.use_nodes = True

nodes = mat.node_tree.nodes
bsdf = nodes.get("Principled BSDF")

tex_node = nodes.new(type="ShaderNodeTexImage")
tex_node.image = bpy.data.images.load(str(texture_path))
tex_node.extension = "CLIP"

mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.75

board.data.materials.append(mat)

# UV düzenle: bütün texture ön yüze otursun.
# Cube olduğu için default UV yeterli değil; hızlı çözüm olarak smart unwrap.
bpy.context.view_layer.objects.active = board
board.select_set(True)
pole.select_set(False)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
bpy.ops.object.mode_set(mode="OBJECT")

# Arka panel gri olsun diye ayrı ince arka plaka ekleyelim.
bpy.ops.mesh.primitive_cube_add(
    size=1.0,
    location=(0.0, 0.018, board_center_z),
)
back = bpy.context.object
back.name = f"{sign_name}_back"
back.dimensions = (board_w, board_t, board_h)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

mat_back = bpy.data.materials.new(f"M_{sign_name}_back")
mat_back.diffuse_color = (0.78, 0.78, 0.78, 1.0)
back.data.materials.append(mat_back)

# Objeleri seç ve tek FBX export et
bpy.ops.object.select_all(action="DESELECT")
pole.select_set(True)
board.select_set(True)
back.select_set(True)
bpy.context.view_layer.objects.active = board

fbx_path.parent.mkdir(parents=True, exist_ok=True)

bpy.ops.export_scene.fbx(
    filepath=str(fbx_path),
    use_selection=True,
    apply_unit_scale=True,
    bake_space_transform=False,
    object_types={"MESH"},
    path_mode="COPY",
    embed_textures=False,
    axis_forward="-Z",
    axis_up="Y",
)

print(f"EXPORTED {fbx_path}")
