#!/usr/bin/env python3
import argparse
import ast
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_XML = (
    PACKAGE_ROOT
    / "linker_hand_mujoco_ros2"
    / "urdf"
    / "L20"
    / "linker_hand_l20_right"
    / "linker_hand_l20_right.xml"
)
# DEFAULT_LAUNCH = PACKAGE_ROOT / "launch" / "linker_hand_mujoco_ros2.launch.py"
DEFAULT_LAUNCH = PACKAGE_ROOT / "launch" / "linker_hand_l20a_contact.launch.py"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "contact_patch_output"

sys.path.insert(0, str(PACKAGE_ROOT))
from linker_hand_mujoco_ros2.utils.mapping import (  # noqa: E402
    L20_JOINT_MAP,
    range_to_arc_right,
)


def parse_float_list(value, default=None):
    if value is None:
        return default if default is not None else []
    return [float(item) for item in value.split()]


def quat_to_matrix(quat):
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=float,
    )


def make_transform(pos=None, quat=None, rotation=None):
    transform = np.eye(4)
    if quat is not None:
        transform[:3, :3] = quat_to_matrix(quat)
    if rotation is not None:
        transform[:3, :3] = transform[:3, :3] @ rotation
    if pos is not None:
        transform[:3, 3] = pos
    return transform


def transform_points(transform, points):
    hom = np.c_[points, np.ones(len(points))]
    return (transform @ hom.T).T[:, :3]


def read_launch_params(launch_path):
    text = Path(launch_path).read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(launch_path))
    params = {}

    def literal(node):
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                key_value = literal(key)
                if key_value in {"hand_type", "hand_joint", "initial_position"}:
                    params[key_value] = literal(value)
    return params


def map_initial_position_to_qpos(position):
    hand_arc = range_to_arc_right(position, "L20")
    mapped = [0.0] * len(L20_JOINT_MAP)
    for target_idx, source_idx in L20_JOINT_MAP.items():
        if source_idx < len(hand_arc):
            mapped[target_idx] = hand_arc[source_idx]
    return np.array(mapped, dtype=float)


def load_stl(path):
    data = Path(path).read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack("<I", data[80:84])[0]
        expected_size = 84 + tri_count * 50
        if expected_size == len(data):
            vertices = []
            faces = []
            offset = 84
            for _ in range(tri_count):
                offset += 12
                face = []
                for _ in range(3):
                    vertex = struct.unpack("<fff", data[offset : offset + 12])
                    offset += 12
                    face.append(len(vertices))
                    vertices.append(vertex)
                faces.append(face)
                offset += 2
            return np.array(vertices, dtype=float), np.array(faces, dtype=np.int32)

    vertices = []
    faces = []
    current_face = []
    for raw_line in data.decode("utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            current_face.append(len(vertices))
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(current_face) == 3:
            faces.append(current_face)
            current_face = []
    if not vertices:
        raise ValueError(f"Could not parse STL: {path}")
    return np.array(vertices, dtype=float), np.array(faces, dtype=np.int32)


def collect_mesh_files(xml_root, xml_path):
    mesh_files = {}
    for mesh in xml_root.findall(".//asset/mesh"):
        name = mesh.get("name")
        file_name = mesh.get("file")
        if name and file_name:
            mesh_files[name] = (Path(xml_path).parent / file_name).resolve()
    return mesh_files


def collect_floor_z(xml_root):
    floor = xml_root.find(".//worldbody/geom[@name='floor']")
    if floor is None:
        return 0.0
    pos = parse_float_list(floor.get("pos"), [0.0, 0.0, 0.0])
    return pos[2]


def local_pose(element):
    pos = np.array(parse_float_list(element.get("pos"), [0.0, 0.0, 0.0]), dtype=float)
    quat = np.array(parse_float_list(element.get("quat"), [1.0, 0.0, 0.0, 0.0]), dtype=float)
    return pos, quat


def build_body_index(xml_root):
    body_by_name = {}
    parent_by_name = {}

    def visit(parent_name, body):
        name = body.get("name")
        if name:
            body_by_name[name] = body
            parent_by_name[name] = parent_name
        for child in body.findall("body"):
            visit(name, child)

    worldbody = xml_root.find("worldbody")
    for body in worldbody.findall("body"):
        visit(None, body)
    return body_by_name, parent_by_name


def body_chain(body_name, parent_by_name):
    chain = []
    current = body_name
    while current is not None:
        chain.append(current)
        current = parent_by_name[current]
    return list(reversed(chain))


def compute_body_transform(body_name, body_by_name, parent_by_name, qpos_by_joint):
    transform = np.eye(4)
    for name in body_chain(body_name, parent_by_name):
        body = body_by_name[name]
        pos, quat = local_pose(body)
        transform = transform @ make_transform(pos, quat)

        joint = body.find("joint")
        if joint is not None:
            joint_name = joint.get("name")
            angle = qpos_by_joint.get(joint_name, 0.0)
            axis = parse_float_list(joint.get("axis"), [0.0, 0.0, 1.0])
            joint_pos = np.array(parse_float_list(joint.get("pos"), [0.0, 0.0, 0.0]), dtype=float)
            transform = (
                transform
                @ make_transform(joint_pos)
                @ make_transform(rotation=axis_angle_to_matrix(axis, angle))
                @ make_transform(-joint_pos)
            )
    return transform


def collect_joint_order(xml_root):
    return [joint.get("name") for joint in xml_root.findall(".//joint") if joint.get("name")]


def find_geom_for_mesh(body, mesh_name):
    for geom in body.findall("geom"):
        if geom.get("mesh") == mesh_name:
            return geom
    raise ValueError(f"Body {body.get('name')} has no geom with mesh={mesh_name}")


def write_ply(path, vertices, faces, selected_faces):
    selected_faces = set(int(item) for item in selected_faces)
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment contact patch visualization\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for vertex in vertices:
            f.write(f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for idx, face in enumerate(faces):
            if idx in selected_faces:
                rgba = "230 40 30 255"
            else:
                rgba = "190 190 190 70"
            f.write(f"3 {face[0]} {face[1]} {face[2]} {rgba}\n")


def write_png(path, vertices, faces, selected_faces, lowest_point, floor_z):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception as exc:
        print(f"Skip PNG because matplotlib is unavailable: {exc}")
        return

    selected_faces = np.array(sorted(selected_faces), dtype=np.int32)
    rest_faces = np.setdiff1d(np.arange(len(faces)), selected_faces)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if len(rest_faces):
        rest_mesh = Poly3DCollection(vertices[faces[rest_faces]], alpha=0.12)
        rest_mesh.set_facecolor((0.6, 0.6, 0.6, 0.12))
        rest_mesh.set_edgecolor((0.3, 0.3, 0.3, 0.05))
        ax.add_collection3d(rest_mesh)

    if len(selected_faces):
        patch_mesh = Poly3DCollection(vertices[faces[selected_faces]], alpha=0.95)
        patch_mesh.set_facecolor((0.9, 0.05, 0.02, 0.95))
        patch_mesh.set_edgecolor((0.5, 0.02, 0.01, 0.4))
        ax.add_collection3d(patch_mesh)

    ax.scatter(
        [lowest_point[0]],
        [lowest_point[1]],
        [lowest_point[2]],
        color="blue",
        s=40,
        label="lowest point",
    )

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    span = max(maxs - mins)
    center = (mins + maxs) / 2
    for setter, coord in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        setter(center[coord] - span / 2, center[coord] + span / 2)
    ax.plot(
        [mins[0], maxs[0]],
        [mins[1], mins[1]],
        [floor_z, floor_z],
        color="black",
        linewidth=1,
        alpha=0.5,
    )
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_zlabel("world z (m)")
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_svg(path, vertices, faces, selected_faces, lowest_point, floor_z):
    selected_faces = set(int(item) for item in selected_faces)
    side_bounds = np.array(
        [
            [vertices[:, 0].min(), min(vertices[:, 2].min(), floor_z)],
            [vertices[:, 0].max(), vertices[:, 2].max()],
        ],
        dtype=float,
    )
    top_bounds = np.array(
        [
            [vertices[:, 0].min(), vertices[:, 1].min()],
            [vertices[:, 0].max(), vertices[:, 1].max()],
        ],
        dtype=float,
    )

    width = 1200
    height = 620
    margin = 52
    panel_gap = 56
    panel_w = (width - 2 * margin - panel_gap) / 2
    panel_h = height - 2 * margin

    def projector(bounds, origin_x):
        mins, maxs = bounds
        span = np.maximum(maxs - mins, 1e-9)
        scale = min(panel_w / span[0], panel_h / span[1]) * 0.92
        offset_x = origin_x + panel_w / 2 - (mins[0] + maxs[0]) * 0.5 * scale
        offset_y = margin + panel_h / 2 + (mins[1] + maxs[1]) * 0.5 * scale

        def project(point2):
            x = offset_x + point2[0] * scale
            y = offset_y - point2[1] * scale
            return x, y

        return project

    side_project = projector(side_bounds, margin)
    top_project = projector(top_bounds, margin + panel_w + panel_gap)

    def polygon(points, project, color, opacity, stroke="none", stroke_width=0.0):
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in [project(point) for point in points])
        return (
            f'<polygon points="{coords}" fill="{color}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="52" y="32" font-family="sans-serif" font-size="18" fill="#222">'
        "Side view: x/z, contact patch around lowest point</text>",
        f'<text x="{margin + panel_w + panel_gap:.0f}" y="32" '
        'font-family="sans-serif" font-size="18" fill="#222">Top view: x/y</text>',
    ]

    rest_faces = [idx for idx in range(len(faces)) if idx not in selected_faces]
    for idx in rest_faces[:: max(1, len(rest_faces) // 4000)]:
        tri = vertices[faces[idx]]
        elements.append(polygon(tri[:, [0, 2]], side_project, "#b8b8b8", 0.16))
        elements.append(polygon(tri[:, [0, 1]], top_project, "#b8b8b8", 0.16))

    for idx in sorted(selected_faces):
        tri = vertices[faces[idx]]
        elements.append(polygon(tri[:, [0, 2]], side_project, "#e62d22", 0.78, "#8a1812", 0.25))
        elements.append(polygon(tri[:, [0, 1]], top_project, "#e62d22", 0.78, "#8a1812", 0.25))

    x0, y0 = side_project([side_bounds[0, 0], floor_z])
    x1, y1 = side_project([side_bounds[1, 0], floor_z])
    elements.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="#222" stroke-width="1.5"/>')
    elements.append(f'<text x="{x0:.2f}" y="{y0 - 6:.2f}" font-family="sans-serif" font-size="12" fill="#222">floor z={floor_z:.3f}m</text>')

    lx, ly = side_project(lowest_point[[0, 2]])
    elements.append(f'<circle cx="{lx:.2f}" cy="{ly:.2f}" r="5" fill="#155bd5" stroke="white" stroke-width="1.5"/>')
    tx, ty = top_project(lowest_point[[0, 1]])
    elements.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="5" fill="#155bd5" stroke="white" stroke-width="1.5"/>')
    elements.append("</svg>\n")
    Path(path).write_text("\n".join(elements), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the contact patch around the lowest point of index_link3 in the launch initial pose."
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--launch", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--body", default="index_link3")
    parser.add_argument("--mesh", default="index_link3")
    parser.add_argument("--radius", type=float, default=0.004, help="Patch radius in meters.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    launch_params = read_launch_params(args.launch)
    initial_position = launch_params.get("initial_position")
    if not initial_position:
        raise ValueError(f"No initial_position found in launch file: {args.launch}")
    if launch_params.get("hand_joint") != "L20" or launch_params.get("hand_type") != "right":
        raise ValueError("This script currently expects launch hand_joint='L20' and hand_type='right'.")

    qpos = map_initial_position_to_qpos(initial_position)
    xml_root = ET.parse(args.xml).getroot()
    mesh_files = collect_mesh_files(xml_root, args.xml)
    floor_z = collect_floor_z(xml_root)
    body_by_name, parent_by_name = build_body_index(xml_root)
    joint_order = collect_joint_order(xml_root)
    qpos_by_joint = {name: qpos[idx] for idx, name in enumerate(joint_order) if idx < len(qpos)}

    body = body_by_name[args.body]
    body_transform = compute_body_transform(args.body, body_by_name, parent_by_name, qpos_by_joint)
    geom = find_geom_for_mesh(body, args.mesh)
    geom_pos, geom_quat = local_pose(geom)
    geom_transform = body_transform @ make_transform(geom_pos, geom_quat)

    local_vertices, faces = load_stl(mesh_files[args.mesh])
    world_vertices = transform_points(geom_transform, local_vertices)
    lowest_idx = int(np.argmin(world_vertices[:, 2]))
    lowest_point = world_vertices[lowest_idx]

    centroids = world_vertices[faces].mean(axis=1)
    distances = np.linalg.norm(centroids - lowest_point, axis=1)
    selected_faces = set(np.where(distances <= args.radius)[0].tolist())
    if not selected_faces:
        selected_faces.add(int(np.argmin(distances)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = args.output_dir / f"{args.body}_contact_patch.ply"
    png_path = args.output_dir / f"{args.body}_contact_patch.png"
    svg_path = args.output_dir / f"{args.body}_contact_patch.svg"
    write_ply(ply_path, world_vertices, faces, selected_faces)
    write_svg(svg_path, world_vertices, faces, selected_faces, lowest_point, floor_z)
    write_png(png_path, world_vertices, faces, selected_faces, lowest_point, floor_z)

    print(f"body: {args.body}")
    print(f"mesh: {args.mesh}")
    print(f"radius_m: {args.radius}")
    print(f"floor_z_m: {floor_z:.9g}")
    print(
        "lowest_point_world_xyz_m: "
        f"{lowest_point[0]:.9g} {lowest_point[1]:.9g} {lowest_point[2]:.9g}"
    )
    print(f"lowest_point_height_above_floor_m: {lowest_point[2] - floor_z:.9g}")
    print(f"selected_faces: {len(selected_faces)} / {len(faces)}")
    print(f"ply: {ply_path}")
    print(f"svg: {svg_path}")
    print(f"png: {png_path}")


if __name__ == "__main__":
    main()
