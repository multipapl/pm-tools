"""Restore Rotation: give a mesh with applied rotation its rotation back.

The mesh is turned to a clean local orientation and the object receives the
opposite rotation, so nothing moves in the viewport, but clearing the
rotation (Alt+R) now stands the object straight. The orientation comes from:

- Auto: the tightest box around each selected object.
- Pick Face: the clicked face (and the flat area around it) becomes the bottom.
- Edit Mode: the selected faces become the bottom.

Of the orientations that fit, the one closest to the current look is used,
so the restored rotation stays as small as possible.
"""

import itertools
import math

import bpy
import gpu
import numpy as np
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

UI_CATEGORY = "SCENE_MANAGEMENT"

# --- LOGIC ---

EPSILON = 1e-9
UNIFORM_SCALE_TOLERANCE = 1e-5
SNAP_TOLERANCE = 1e-6
# Auto: face normals are binned (~0.6 degrees) and the heaviest bins are tried as box sides.
NORMAL_BIN_SCALE = 100.0
AUTO_CANDIDATES = 24
# Boxes this close in volume count as equal, the one needing less rotation wins.
VOLUME_TOLERANCE = 1e-4
FLAT_BOX_PADDING = 1e-3
RECT_CHUNK = 256

FILL_COLOR = (1.0, 0.55, 0.1, 0.3)
OUTLINE_COLOR = (1.0, 1.0, 1.0, 0.9)
AXIS_COLORS = ((0.96, 0.2, 0.32, 1.0), (0.54, 0.86, 0.0, 1.0), (0.16, 0.56, 1.0, 1.0))


def _quarter_turn(steps):
    c, s = ((1, 0), (0, 1), (-1, 0), (0, -1))[steps % 4]
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)


def _axis_rotations():
    """The 24 rotations that map the coordinate axes onto each other."""
    rotations = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            matrix[list(order), [0, 1, 2]] = signs
            if np.linalg.det(matrix) > 0.0:
                rotations.append(matrix)
    return rotations


QUARTER_TURNS = [_quarter_turn(steps) for steps in range(4)]
AXIS_ROTATIONS = _axis_rotations()


def _read(collection, attribute, dtype, width=1):
    data = np.empty(len(collection) * width, dtype=dtype)
    collection.foreach_get(attribute, data)
    return data.reshape(-1, width) if width > 1 else data


def _ranges(starts, counts):
    """Concatenated index ranges [start, start + count) as one array."""
    shifts = np.repeat(starts - np.cumsum(counts) + counts, counts)
    return shifts + np.arange(int(counts.sum()))


def _bake_matrix(basis):
    """The part of basis the mesh absorbs: only the rotation while the scale
    is uniform (the scale then stays on the object), the whole 3x3 otherwise."""
    matrix = basis.to_3x3()
    lengths = [column.length for column in matrix.col]
    if min(lengths) <= EPSILON:
        return None, False
    if max(lengths) - min(lengths) > UNIFORM_SCALE_TOLERANCE * max(lengths):
        return matrix, False
    rotation = matrix.normalized()
    if rotation.determinant() < 0.0:
        rotation.negate()
    return rotation, True


def _editable_mesh(obj):
    return obj is not None and obj.type == 'MESH' and obj.is_editable and obj.data.is_editable


class _Geometry:
    """Mesh of one object measured in 'bake space': local coordinates with
    the part of the object's rotation the mesh will absorb already applied,
    so bake-space axes are the axes the object is aligned to now."""

    def __init__(self, obj, bake, uniform):
        self.obj = obj
        self.bake = bake
        self.uniform = uniform
        self.matrix_world = obj.matrix_world.copy()
        mesh = obj.data
        self.local = _read(mesh.vertices, "co", np.float32, 3).astype(np.float64)
        self.loop_vert = _read(mesh.loops, "vertex_index", np.int32).astype(np.int64)
        self.poly_start = _read(mesh.polygons, "loop_start", np.int32).astype(np.int64)
        self.poly_total = _read(mesh.polygons, "loop_total", np.int32).astype(np.int64)
        self.points = self.local @ np.array(bake).T

        # Newell's method: area-weighted normals that hold up for n-gons too.
        loop_next = np.arange(1, len(self.loop_vert) + 1)
        loop_next[self.poly_start + self.poly_total - 1] = self.poly_start
        crosses = np.cross(self.points[self.loop_vert], self.points[self.loop_vert[loop_next]])
        self.area_normals = np.add.reduceat(crosses, self.poly_start, axis=0) * 0.5
        self.areas = np.linalg.norm(self.area_normals, axis=1)
        self.normals = self.area_normals / np.maximum(self.areas, EPSILON)[:, None]

        self._bvh = None
        self._edge_faces = None
        self.tri_verts = None
        self.tri_poly = None

    @staticmethod
    def load(obj):
        bake, uniform = _bake_matrix(obj.matrix_basis)
        if bake is None or not obj.data.polygons:
            return None
        return _Geometry(obj, bake, uniform)

    def face_verts(self, faces):
        return np.unique(self.loop_vert[_ranges(self.poly_start[faces], self.poly_total[faces])])

    def bvh(self):
        if self._bvh is None:
            mesh = self.obj.data
            mesh.calc_loop_triangles()
            self.tri_verts = _read(mesh.loop_triangles, "vertices", np.int32, 3)
            self.tri_poly = _read(mesh.loop_triangles, "polygon_index", np.int32)
            self._bvh = BVHTree.FromPolygons(self.local.tolist(), self.tri_verts.tolist(), all_triangles=True)
        return self._bvh

    def edge_faces(self):
        """Loop edges plus a CSR table edge -> faces."""
        if self._edge_faces is None:
            mesh = self.obj.data
            loop_edge = _read(mesh.loops, "edge_index", np.int32).astype(np.int64)
            loop_face = np.repeat(np.arange(len(self.poly_start)), self.poly_total)
            order = np.argsort(loop_edge, kind='stable')
            starts = np.searchsorted(loop_edge[order], np.arange(len(mesh.edges) + 1))
            self._edge_faces = (loop_edge, loop_face[order], starts)
        return self._edge_faces


# -- Planar fitting -----------------------------------------------------------

def _plane_basis(up):
    u = np.array(Vector(up).orthogonal().normalized())
    return u, np.cross(up, u)


def _discard_interior(points):
    """Akl-Toussaint: drop points inside the octagon of extreme points."""
    x, y = points[:, 0], points[:, 1]
    octagon = points[[int(np.argmax(key)) for key in (x, x + y, y, y - x, -x, -x - y, -y, x - y)]]
    inside = np.ones(len(points), dtype=bool)
    sides = 0
    for start, end in zip(octagon, np.roll(octagon, -1, axis=0)):
        if np.array_equal(start, end):
            continue
        sides += 1
        inside &= (end[0] - start[0]) * (y - start[1]) - (end[1] - start[1]) * (x - start[0]) > EPSILON
    return points if sides < 3 else points[~inside]


def _convex_hull(points):
    """Monotone chain, counter-clockwise, without collinear points."""
    if len(points) > 64:
        points = _discard_interior(points)
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return points
    ordered = points.tolist()

    def half(sequence):
        chain = []
        for point in sequence:
            while len(chain) >= 2 and (
                (chain[-1][0] - chain[-2][0]) * (point[1] - chain[-2][1])
                - (chain[-1][1] - chain[-2][1]) * (point[0] - chain[-2][0])
            ) <= 0.0:
                chain.pop()
            chain.append(point)
        return chain

    return np.array(half(ordered)[:-1] + half(reversed(ordered))[:-1])


def _min_area_rect(points):
    """Smallest rectangle around 2D points: (center, unit axis, half sizes, area).
    One of its sides always lies on a convex hull edge."""
    hull = _convex_hull(points)
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    directions = edges[lengths > EPSILON] / lengths[lengths > EPSILON, None]
    if not len(directions):
        directions = np.array(((1.0, 0.0),))

    best_area, best_axis = math.inf, directions[0]
    for start in range(0, len(directions), RECT_CHUNK):
        chunk = directions[start:start + RECT_CHUNK]
        along = hull @ chunk.T
        across = hull @ np.column_stack((-chunk[:, 1], chunk[:, 0])).T
        areas = np.ptp(along, axis=0) * np.ptp(across, axis=0)
        index = int(np.argmin(areas))
        if areas[index] < best_area:
            best_area, best_axis = float(areas[index]), chunk[index]

    normal = np.array((-best_axis[1], best_axis[0]))
    along, across = hull @ best_axis, hull @ normal
    center = best_axis * (along.max() + along.min()) * 0.5 + normal * (across.max() + across.min()) * 0.5
    half = np.array((np.ptp(along), np.ptp(across))) * 0.5
    return center, best_axis, half, best_area


def _frame(up, side, turns):
    """Rotation whose columns are the new X, Y, Z in bake space: Z = up,
    X along side or across it, whichever needs the least rotation."""
    axes = np.column_stack((side, np.cross(up, side), up))
    best = max(QUARTER_TURNS, key=lambda turn: np.trace(axes @ turn))
    return axes @ best @ QUARTER_TURNS[turns]


class _FaceFit:
    """Bottom plane of a face region and the rectangle it is aligned to."""

    __slots__ = ("faces", "up", "side", "corners", "middle", "size")

    def __init__(self, faces, up, side, corners, middle, size):
        self.faces = faces
        self.up = up
        self.side = side
        self.corners = corners
        self.middle = middle
        self.size = size


def _face_fit(geometry, faces, whole_object, upside_down):
    normal = geometry.area_normals[faces].sum(axis=0)
    length = np.linalg.norm(normal)
    if length <= EPSILON:
        return None
    # The faces are the bottom, so their normal points down.
    up = normal / length * (1.0 if upside_down else -1.0)
    u, w = _plane_basis(up)
    region = geometry.points[geometry.face_verts(faces)]
    points = geometry.points if whole_object else region
    origin = region.mean(axis=0)
    offsets = points - origin
    center, axis, half, _area = _min_area_rect(np.column_stack((offsets @ u, offsets @ w)))
    side = axis[0] * u + axis[1] * w
    across = np.cross(up, side)
    middle = origin + center[0] * u + center[1] * w
    corners = np.array([
        middle + sx * half[0] * side + sy * half[1] * across
        for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1))
    ])
    return _FaceFit(faces, up, side, corners, middle, float(max(half)))


def _flat_region(geometry, seed, max_angle):
    """Faces connected to seed whose normals stay within max_angle of it."""
    loop_edge, edge_faces, edge_start = geometry.edge_faces()
    similar = geometry.normals @ geometry.normals[seed] >= math.cos(max_angle) - EPSILON
    inside = np.zeros(len(similar), dtype=bool)
    inside[seed] = True
    frontier = np.array((seed,))
    while frontier.size:
        edges = loop_edge[_ranges(geometry.poly_start[frontier], geometry.poly_total[frontier])]
        starts = edge_start[edges]
        faces = np.unique(edge_faces[_ranges(starts, edge_start[edges + 1] - starts)])
        frontier = faces[similar[faces] & ~inside[faces]]
        inside[frontier] = True
    return np.flatnonzero(inside)


def _auto_candidates(geometry):
    """Directions worth trying as a box side: the current axes and the
    normal directions that carry the most surface area."""
    valid = geometry.areas > EPSILON
    normals, areas = geometry.normals[valid], geometry.areas[valid]
    candidates = [np.eye(3)]
    if len(normals):
        # n and -n are the same box side: flip so the largest component is positive.
        dominant = np.argmax(np.abs(normals), axis=1)
        normals = normals * np.sign(normals[np.arange(len(normals)), dominant])[:, None]
        keys = np.round(normals * NORMAL_BIN_SCALE).astype(np.int64)
        _keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        weights = np.bincount(inverse, weights=areas)
        sums = np.zeros((len(weights), 3))
        np.add.at(sums, inverse, normals * areas[:, None])
        top = np.argsort(weights)[::-1][:AUTO_CANDIDATES]
        directions = sums[top]
        candidates.append(directions / np.maximum(np.linalg.norm(directions, axis=1), EPSILON)[:, None])
    return np.vstack(candidates)


def _auto_frame(geometry, upside_down, turns):
    """Orientation of the tightest box around the mesh."""
    points = geometry.points - geometry.points.mean(axis=0)
    padding = FLAT_BOX_PADDING * max(float(np.ptp(points, axis=0).max()), EPSILON)
    boxes = []
    for normal in _auto_candidates(geometry):
        u, w = _plane_basis(normal)
        _center, axis, half, _area = _min_area_rect(np.column_stack((points @ u, points @ w)))
        height = float(np.ptp(points @ normal))
        # Padding keeps flat objects comparable: they are all zero volume otherwise.
        volume = (2.0 * half[0] + padding) * (2.0 * half[1] + padding) * (height + padding)
        side = axis[0] * u + axis[1] * w
        axes = np.column_stack((side, np.cross(normal, side), normal))
        boxes.append((volume, max((axes @ rotation for rotation in AXIS_ROTATIONS), key=np.trace)))

    smallest = min(volume for volume, _rotation in boxes)
    rotation = max(
        (rotation for volume, rotation in boxes if volume <= smallest * (1.0 + VOLUME_TOLERANCE)),
        key=np.trace,
    )
    up = -rotation[:, 2] if upside_down else rotation[:, 2]
    return _frame(up, rotation[:, 0], turns)


# -- Applying -------------------------------------------------------------------

def _snap(rotation):
    """Clean float noise so aligned axes come out as exact 0/90 degrees."""
    rotation = rotation.copy()
    rotation[np.abs(rotation) < SNAP_TOLERANCE] = 0.0
    near_one = np.abs(np.abs(rotation) - 1.0) < SNAP_TOLERANCE
    rotation[near_one] = np.sign(rotation[near_one])
    return Matrix(rotation.tolist()).to_quaternion().to_matrix()


def _restore(geometry, rotation, origin):
    """Turn the mesh into the frame given by rotation and give its object(s)
    the inverse, so they stay in place. Returns True if the mesh had to be
    made single user."""
    obj = geometry.obj
    turn = _snap(rotation).transposed() @ geometry.bake
    offset = Vector()
    if origin != 'KEEP':
        local = geometry.local @ np.array(turn).T
        low, high = local.min(axis=0), local.max(axis=0)
        offset = Vector((low + high) * 0.5)
        if origin == 'BOTTOM':
            offset.z = float(low[2])
    mesh_matrix = Matrix.Translation(-offset) @ turn.to_4x4()
    compensation = mesh_matrix.inverted()

    mesh = obj.data
    users = [obj]
    copied = False
    if mesh.users > 1:
        others = [other for other in bpy.data.objects if other.data == mesh and other != obj]
        # Instances with non-uniform scale cannot absorb the rotation exactly.
        if others and (not geometry.uniform or not all(_bake_matrix(other.matrix_basis)[1] for other in others)):
            obj.data = mesh = mesh.copy()
            copied = True
        else:
            users.extend(others)

    mesh.transform(mesh_matrix, shape_keys=True)
    if mesh_matrix.determinant() < 0.0:
        mesh.flip_normals()
    mesh.update()

    for user in users:
        user.matrix_basis = user.matrix_basis @ compensation
        for child in user.children:
            if child.parent_type == 'OBJECT':
                child.matrix_parent_inverse = mesh_matrix @ child.matrix_parent_inverse
    return copied


# -- Picking --------------------------------------------------------------------

def _region_at(screen, x, y):
    """3D Viewport main region under window coordinates (not under a sidebar)."""
    for area in screen.areas:
        if area.type != 'VIEW_3D' or not (area.x <= x < area.x + area.width and area.y <= y < area.y + area.height):
            continue
        window = None
        for region in area.regions:
            if not (region.x <= x < region.x + region.width and region.y <= y < region.y + region.height):
                continue
            if region.type == 'WINDOW':
                window = region
            elif region.width > 1 and region.height > 1:
                return None
        return window
    return None


def _world_bounds(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    low = Vector([min(corner[axis] for corner in corners) for axis in range(3)])
    high = Vector([max(corner[axis] for corner in corners) for axis in range(3)])
    padding = max((high - low).length * 1e-4, 1e-6)
    return low - Vector((padding,) * 3), high + Vector((padding,) * 3)


def _ray_hits_box(origin, direction, low, high):
    near, far = -math.inf, math.inf
    for start, step, lower, upper in zip(origin, direction, low, high):
        if abs(step) < EPSILON:
            if not lower <= start <= upper:
                return False
            continue
        first, second = (lower - start) / step, (upper - start) / step
        near, far = max(near, min(first, second)), min(far, max(first, second))
        if near > far:
            return False
    return far >= 0.0


def _ray_cast(geometry, origin, direction):
    """(distance, face, world location) of the ray against the mesh, or None."""
    inverse = geometry.matrix_world.inverted_safe()
    local_direction = inverse.to_3x3() @ direction
    if local_direction.length <= EPSILON:
        return None
    location, _normal, index, _distance = geometry.bvh().ray_cast(inverse @ origin, local_direction.normalized())
    if location is None:
        return None
    world = geometry.matrix_world @ location
    return (world - origin).length, int(geometry.tri_poly[index]), world


def _to_world(points, matrix):
    return (points @ np.array(matrix.to_3x3()).T + np.array(matrix.translation)).astype(np.float32)


class _Preview:
    """What the draw callback shows: the picked region and the new axes."""

    def __init__(self):
        self.fill = None
        self.fill_batch = None
        self.lines = []

    def set_fill(self, coords):
        self.fill = coords
        self.fill_batch = None

    def clear(self):
        self.set_fill(None)
        self.lines = []


def _draw_preview(preview):
    if preview.fill is None and not preview.lines:
        return
    depth_test = gpu.state.depth_test_get()
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('ALPHA')
    if preview.fill is not None and len(preview.fill):
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        if preview.fill_batch is None:
            preview.fill_batch = batch_for_shader(shader, 'TRIS', {"pos": preview.fill})
        shader.uniform_float("color", FILL_COLOR)
        preview.fill_batch.draw(shader)
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    for coords, color, width in preview.lines:
        shader.uniform_float("color", color)
        shader.uniform_float("lineWidth", width)
        batch_for_shader(shader, 'LINES', {"pos": coords}).draw(shader)
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set(depth_test)


def _draw_status(header, _context):
    layout = header.layout
    layout.label(text="Pick Bottom Face", icon='MOUSE_LMB')
    row = layout.row(align=True)
    row.label(text="", icon='EVENT_SHIFT')
    row.label(text="Pick and Continue", icon='MOUSE_LMB')
    layout.label(text="Finish", icon='MOUSE_RMB')
    layout.label(text="Navigate", icon='MOUSE_MMB')


def _redraw_viewports(context):
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


NAVIGATION_EVENTS = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE',
    'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE', 'MOUSESMARTZOOM',
}


# --- OPERATORS ---

class PM_RestoreRotationPick(bpy.types.PropertyGroup):
    object_name: bpy.props.StringProperty()  # type: ignore
    face: bpy.props.IntProperty()  # type: ignore


class PM_OT_RestoreRotation(bpy.types.Operator):
    """Give a mesh with applied rotation its rotation back. The object stays in place,
but gets a local rotation, so clearing it (Alt+R) stands the object straight"""
    bl_idname = "pm.restore_rotation"
    bl_label = "Restore Rotation"
    bl_options = {'REGISTER', 'UNDO'}

    source: bpy.props.EnumProperty(  # type: ignore
        name="Source",
        items=[
            ('AUTO', "Auto", "Fit the tightest box around each selected object"),
            ('PICK', "Pick Face", "Click the face that should become the bottom"),
            ('SELECTED', "Selected Faces", "The selected faces become the bottom"),
        ],
        default='AUTO',
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    picks: bpy.props.CollectionProperty(type=PM_RestoreRotationPick, options={'HIDDEN', 'SKIP_SAVE'})  # type: ignore
    flat_angle: bpy.props.FloatProperty(  # type: ignore
        name="Flat Angle",
        description="Faces around the picked one that bend less than this count as the same bottom",
        default=math.radians(5.0),
        min=0.0,
        max=math.radians(45.0),
        subtype='ANGLE',
    )
    fit: bpy.props.EnumProperty(  # type: ignore
        name="Align Sides To",
        description="What the X and Y axes are lined up with",
        items=[
            ('FACE', "Bottom Face", "Sides of the bottom face"),
            ('OBJECT', "Whole Object", "Footprint of the whole object on the bottom plane"),
        ],
        default='FACE',
    )
    turn: bpy.props.EnumProperty(  # type: ignore
        name="Turn",
        description="Extra turn of the local axes around Z",
        items=[('0', "0°", ""), ('1', "90°", ""), ('2', "180°", ""), ('3', "270°", "")],
        default='0',
    )
    upside_down: bpy.props.BoolProperty(  # type: ignore
        name="Upside Down",
        description="Flip the result: with a picked face, that face becomes the top",
        default=False,
    )
    origin: bpy.props.EnumProperty(  # type: ignore
        name="Origin",
        items=[
            ('KEEP', "Keep", "Leave the origin where it is"),
            ('BOTTOM', "Bottom Center", "Bottom center of the aligned bounding box"),
            ('CENTER', "Center", "Center of the aligned bounding box"),
        ],
        default='KEEP',
    )

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_MESH'}

    @classmethod
    def description(cls, context, properties):
        source = properties.source
        if source == 'PICK':
            return ("Click the face that should become the bottom (Shift+Click to keep picking).\n"
                    "If objects are selected, only they can be picked, even behind other objects")
        if source == 'SELECTED':
            return "Align each edited mesh so its selected faces become the bottom"
        return "Align each selected object to the tightest box around it"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        if self.source == 'PICK':
            layout.prop(self, "flat_angle")
        if self.source != 'AUTO':
            layout.prop(self, "fit")
        layout.row().prop(self, "turn", expand=True)
        layout.prop(self, "upside_down")
        layout.prop(self, "origin")

    # -- Execution ---------------------------------------------------------------

    def _restore_faces(self, geometry, faces):
        fit = _face_fit(geometry, faces, self.fit == 'OBJECT', self.upside_down)
        if fit is None:
            return None
        return _restore(geometry, _frame(fit.up, fit.side, int(self.turn)), self.origin)

    def _apply_pick(self, pick):
        """Restore one pick. Returns None if it could not be used, else whether
        the mesh was made single user."""
        obj = bpy.data.objects.get(pick.object_name)
        if not _editable_mesh(obj):
            return None
        geometry = _Geometry.load(obj)
        if geometry is None or not 0 <= pick.face < len(geometry.poly_start):
            return None
        return self._restore_faces(geometry, _flat_region(geometry, pick.face, self.flat_angle))

    def execute(self, context):
        results = []
        if context.mode == 'EDIT_MESH':
            self.source = 'SELECTED'
            objects = [obj for obj in context.objects_in_mode_unique_data if obj.type == 'MESH']
            # Leave Edit Mode: the undo step then holds the object rotation too.
            bpy.ops.object.mode_set(mode='OBJECT')
            for obj in objects:
                faces = np.flatnonzero(_read(obj.data.polygons, "select", bool))
                geometry = _Geometry.load(obj) if faces.size and _editable_mesh(obj) else None
                results.append(None if geometry is None else self._restore_faces(geometry, faces))
        elif self.source == 'PICK':
            results = [self._apply_pick(pick) for pick in self.picks]
        else:
            for obj in context.selected_objects:
                geometry = _Geometry.load(obj) if _editable_mesh(obj) else None
                if geometry is None:
                    continue
                rotation = _auto_frame(geometry, self.upside_down, int(self.turn))
                results.append(_restore(geometry, rotation, self.origin))
        return self._report(results)

    def _report(self, results):
        restored = [copied for copied in results if copied is not None]
        if not restored:
            message = {
                'SELECTED': "Select the faces that should become the bottom",
                'PICK': "Nothing was picked",
            }.get(self.source, "Select mesh objects")
            self.report({'WARNING'}, message)
            return {'CANCELLED'}
        message = f"Restored rotation of {len(restored)} object{'s' if len(restored) != 1 else ''}"
        if any(restored):
            message += f" ({sum(restored)} shared meshes made single user)"
        if len(restored) < len(results):
            message += f", {len(results) - len(restored)} skipped"
        self.report({'INFO'}, message)
        return {'FINISHED'}

    # -- Picking -----------------------------------------------------------------

    def invoke(self, context, event):
        if context.mode == 'EDIT_MESH' or self.source != 'PICK':
            return self.execute(context)
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Run Pick Face from the 3D Viewport")
            return {'CANCELLED'}

        self.picks.clear()
        self._results = []
        self._geometries = {}
        self._bounds = {}
        self._fit = None
        self._hover = None
        self._preview = _Preview()
        self._handle = bpy.types.SpaceView3D.draw_handler_add(_draw_preview, (self._preview,), 'WINDOW', 'POST_VIEW')
        self._area = context.area
        context.window.cursor_modal_set('EYEDROPPER')
        context.workspace.status_text_set(_draw_status)
        self._area.header_text_set("Restore Rotation: click the face that should become the bottom")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception:
            self._finish(context)
            raise

    def _modal(self, context, event):
        if (
            event.type in NAVIGATION_EVENTS
            or event.type.startswith(('NDOF', 'NUMPAD'))
            or (event.type == 'LEFTMOUSE' and event.alt)
        ):
            return {'PASS_THROUGH'}
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._update_hover(context, event)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS' and self._hover is not None:
            pick = self.picks.add()
            pick.object_name, pick.face = self._hover
            self._results.append(self._apply_pick(pick))
            # The picked mesh (and its instances) changed: measure again.
            self._geometries.clear()
            self._fit = None
            if not event.shift:
                self._finish(context)
                return self._report(self._results)
            self._update_hover(context, event)
        elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            self._finish(context)
            return self._report(self._results) if len(self.picks) else {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        if self._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)
        self._area.header_text_set(None)
        _redraw_viewports(context)

    def _geometry(self, obj):
        if obj.name not in self._geometries:
            self._geometries[obj.name] = _Geometry.load(obj)
        return self._geometries[obj.name]

    def _cast(self, context, event):
        """(distance, geometry, face, location) under the mouse, or None."""
        region = _region_at(context.screen, event.mouse_x, event.mouse_y)
        if region is None or region.data is None:
            return None
        coord = (event.mouse_x - region.x, event.mouse_y - region.y)
        origin = view3d_utils.region_2d_to_origin_3d(region, region.data, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, region.data, coord)

        # Selected objects can be picked through anything in front of them.
        candidates = [obj for obj in context.selected_objects if _editable_mesh(obj) and obj.visible_get()]
        fallback = None
        if not candidates:
            depsgraph = context.evaluated_depsgraph_get()
            found, location, _normal, _index, obj, _matrix = context.scene.ray_cast(depsgraph, origin, direction)
            if not found or obj is None or not _editable_mesh(obj.original):
                return None
            candidates, fallback = [obj.original], location

        best = None
        for obj in candidates:
            if obj.name not in self._bounds:
                self._bounds[obj.name] = _world_bounds(obj)
            if fallback is None and not _ray_hits_box(origin, direction, *self._bounds[obj.name]):
                continue
            geometry = self._geometry(obj)
            if geometry is None:
                continue
            hit = _ray_cast(geometry, origin, direction)
            if hit is None and fallback is not None:
                # Hit geometry made by modifiers: take the nearest real face.
                local = geometry.matrix_world.inverted_safe() @ fallback
                _location, _normal, index, _distance = geometry.bvh().find_nearest(local)
                if index is not None:
                    hit = ((fallback - origin).length, int(geometry.tri_poly[index]), fallback)
            if hit is not None and (best is None or hit[0] < best[0]):
                best = (hit[0], geometry, hit[1], hit[2])
        return best

    def _update_hover(self, context, event):
        hit = self._cast(context, event)
        self._hover = None
        if hit is None:
            self._preview.clear()
            _redraw_viewports(context)
            return

        _distance, geometry, face, _location = hit
        if self._fit is None or self._fit[0] is not geometry or face not in self._fit[2]:
            faces = _flat_region(geometry, face, self.flat_angle)
            fit = _face_fit(geometry, faces, self.fit == 'OBJECT', self.upside_down)
            self._fit = (geometry, fit, set(faces.tolist()))
            tris = geometry.tri_verts[np.isin(geometry.tri_poly, faces)]
            self._preview.set_fill(_to_world(geometry.local[tris.ravel()], geometry.matrix_world))
        fit = self._fit[1]
        if fit is None:
            self._preview.clear()
            _redraw_viewports(context)
            return

        self._hover = (geometry.obj.name, face)
        rotation = _frame(fit.up, fit.side, int(self.turn))
        to_world = geometry.matrix_world @ geometry.bake.inverted_safe().to_4x4()
        corners = _to_world(fit.corners, to_world)
        outline = corners[[0, 1, 1, 2, 2, 3, 3, 0]]
        lines = [(outline, OUTLINE_COLOR, 1.5)]
        for axis, color in enumerate(AXIS_COLORS):
            tip = fit.middle + rotation[:, axis] * fit.size
            lines.append((_to_world(np.array((fit.middle, tip)), to_world), color, 3.0))
        self._preview.lines = lines
        _redraw_viewports(context)


# --- UI DRAWING ---

def draw_ui(layout, context):
    box = layout.box()
    box.label(text="Restore Rotation", icon='ORIENTATION_GIMBAL')
    if context.mode == 'EDIT_MESH':
        box.operator("pm.restore_rotation", text="From Selected Faces", icon='FACESEL').source = 'SELECTED'
        return
    row = box.row(align=True)
    row.operator("pm.restore_rotation", text="Auto", icon='SHADING_BBOX').source = 'AUTO'
    row.operator("pm.restore_rotation", text="Pick Face", icon='EYEDROPPER').source = 'PICK'


# --- REGISTRATION ---

classes = (
    PM_RestoreRotationPick,
    PM_OT_RestoreRotation,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
