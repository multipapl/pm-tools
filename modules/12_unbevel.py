"""Unbevel: collapse bevels/chamfers baked into a mesh back into sharp edges.

Detection works on the selected faces, so "select all + Unbevel" is the
intended workflow. A bevel is recognised as a *strip*: a run of quads whose
long edges (rails) are all parallel, whose normals rotate monotonically from
one main face to the other and whose segments are much narrower than those
main faces. Faces enclosed only by strip ends (corner patches) collapse with
them. Every collapsed vertex is placed on the intersection of the main-face
planes it touches, so the original sharp corner comes back exactly.
"""

import math
from collections import deque

import bmesh
import bpy
import numpy as np

UI_CATEGORY = "SCENE_OPTIMIZATION"

# --- LOGIC ---

# Bevels from Blender and 3ds Max have exactly parallel rails and near-equal
# segment widths, so the tolerances can stay strict.
PARALLEL_COS = math.cos(math.radians(3.0))
SAME_PLANE_COS = math.cos(math.radians(2.0))
SIMILAR_WIDTH_RATIO = 2.5
# Main faces must be this many times wider than a bevel segment. Confident
# tiers go first so their corner patches can veto doubtful strips.
MAIN_FACE_RATIO = 2.0
SCORE_TIERS = (8.0, 4.0)
SCORE_CAP = 16.0
MAX_PLANE_FACES = 64
# Second pass: faces as wide as an already found bevel count as bevel too.
LEARNED_WIDTH_TOLERANCE = 0.03
LEARNED_MAIN_RATIO = 1.0
TURN_EPSILON = math.radians(0.5)
MIN_TOTAL_TURN = math.radians(2.0)
MAX_TOTAL_TURN = math.radians(175.0)
MAX_SEGMENTS = 64
MAX_PATCH_FACES = 256
PLANE_RANK_RATIO = 0.01  # ~1 degree between planes still counts as a crease
EPSILON = 1e-9


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        parent = self.parent
        root = parent.setdefault(item, item)
        while root != parent[root]:
            root = parent[root]
        while item != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[second] = first


class _Run:
    """One cross-section of a bevel: quads between main faces A and B."""

    __slots__ = ("faces", "rails", "main_a", "main_b", "widths", "score", "mode")

    def __init__(self, faces, rails, main_a, main_b, widths, score, mode):
        self.faces = faces
        self.rails = rails
        self.main_a = main_a
        self.main_b = main_b
        self.widths = widths
        self.score = score
        self.mode = mode


def _direction(edge):
    vector = edge.verts[1].co - edge.verts[0].co
    length = vector.length
    return vector / length if length > EPSILON else None


def _midpoint(edge):
    return (edge.verts[0].co + edge.verts[1].co) * 0.5


def _distance_to_line(point, origin, axis):
    offset = point - origin
    return (offset - axis * offset.dot(axis)).length


def _turn(face_a, face_b, axis):
    """Signed rotation from face_a's normal to face_b's normal around axis."""
    normal_a, normal_b = face_a.normal, face_b.normal
    return math.atan2(axis.dot(normal_a.cross(normal_b)), normal_a.dot(normal_b))


def _other_face(edge, face):
    faces = edge.link_faces
    if len(faces) != 2:
        return None
    return faces[1] if faces[0] == face else faces[0]


def _coplanar(face_a, face_b, tolerance):
    if face_a == face_b:
        return True
    if face_a.normal.dot(face_b.normal) < SAME_PLANE_COS:
        return False
    offset = face_b.calc_center_median() - face_a.calc_center_median()
    return abs(face_a.normal.dot(offset)) <= tolerance


class _BevelDetector:
    """Finds bevel strips and corner patches among the selected faces."""

    def __init__(self, bm, max_width):
        self.max_width = max_width
        self.candidates = {f for f in bm.faces if f.select and len(f.verts) == 4}
        self._quad_edges = {}
        self.mode = 'MANUAL' if max_width > 0.0 else 'AUTO'
        self.learned_widths = []
        self.accepted = []
        self.taken = {}  # strip face -> orientation (rails are edges o and o + 2)
        self.run_of = {}
        self.main_faces = set()
        self.patches = []
        self.patch_faces = set()
        self.skipped = 0

    def quad_edges(self, face):
        edges = self._quad_edges.get(face)
        if edges is None:
            edges = self._quad_edges[face] = tuple(loop.edge for loop in face.loops)
        return edges

    def cross_edges(self, face, orientation):
        edges = self.quad_edges(face)
        return edges[orientation + 1], edges[(orientation + 3) % 4]

    def _quad_width(self, face, orientation):
        edges = self.quad_edges(face)
        axis = _direction(edges[orientation])
        if axis is None:
            return 0.0
        return _distance_to_line(_midpoint(edges[orientation + 2]), edges[orientation].verts[0].co, axis)

    # -- Cross-sections ------------------------------------------------------

    def _accepts_width(self, width, widths, total_width, mode):
        if width <= EPSILON:
            return False
        if mode == 'MANUAL':
            return total_width <= self.max_width
        if mode == 'LEARNED':
            return any(abs(width - known) <= LEARNED_WIDTH_TOLERANCE * known for known in self.learned_widths)
        smallest = min(width, min(widths))
        largest = max(width, max(widths))
        return largest <= SIMILAR_WIDTH_RATIO * smallest

    def _absorbable(self, face):
        return face in self.candidates and face not in self.taken and face not in self.patch_faces

    def _grow(self, seed, orientation, mode):
        """Walk across the strip from seed through its rails until main faces."""
        edges = self.quad_edges(seed)
        first, second = edges[orientation], edges[orientation + 2]
        axis = _direction(first)
        second_axis = _direction(second)
        if axis is None or second_axis is None or abs(axis.dot(second_axis)) < PARALLEL_COS:
            return None
        width = _distance_to_line(_midpoint(second), first.verts[0].co, axis)
        if not self._accepts_width(width, [width], width, mode):
            return None

        faces = deque([seed])
        rails = deque([first, second])
        members = {seed}
        widths = [width]
        inner_turns = []
        sign = 0.0
        mains = [None, None]

        for side in (0, 1):
            while True:
                edge = rails[0] if side == 0 else rails[-1]
                previous = faces[0] if side == 0 else faces[-1]
                following = _other_face(edge, previous)
                if following is None or following in members or len(faces) > MAX_SEGMENTS:
                    # Open boundary, closed ring or too long for a bevel. Still
                    # returned so the caller skips re-walking these faces.
                    return list(faces), list(rails), [None, None], widths, inner_turns, axis

                step = None
                if self._absorbable(following):
                    step = self._step(following, edge, previous, side, axis, rails, widths, sign, mode)
                if step is None:
                    mains[side] = following
                    break

                opposite, step_width, turn = step
                members.add(following)
                widths.append(step_width)
                inner_turns.append(turn)
                if side == 0:
                    faces.appendleft(following)
                    rails.appendleft(opposite)
                else:
                    faces.append(following)
                    rails.append(opposite)
                if not sign and abs(turn) > TURN_EPSILON:
                    sign = math.copysign(1.0, turn)

        return list(faces), list(rails), mains, widths, inner_turns, axis

    def _step(self, face, entry, previous, side, axis, rails, widths, sign, mode):
        edges = self.quad_edges(face)
        opposite = edges[(edges.index(entry) + 2) % 4]
        opposite_axis = _direction(opposite)
        if opposite_axis is None or abs(axis.dot(opposite_axis)) < PARALLEL_COS:
            return None

        width = _distance_to_line(_midpoint(opposite), entry.verts[0].co, axis)
        far_rail = rails[-1] if side == 0 else rails[0]
        total_width = _distance_to_line(_midpoint(opposite), far_rail.verts[0].co, axis)
        if not self._accepts_width(width, widths, total_width, mode):
            return None

        turn = _turn(face, previous, axis) if side == 0 else _turn(previous, face, axis)
        if sign and turn * sign < -TURN_EPSILON:
            return None
        return opposite, width, turn

    def _to_run(self, section, mode):
        faces, rails, (main_a, main_b), widths, inner_turns, axis = section
        if main_a is None or main_b is None or main_a == main_b:
            return None

        turns = [_turn(main_a, faces[0], axis), *inner_turns, _turn(faces[-1], main_b, axis)]
        total = sum(turns)
        if not MIN_TOTAL_TURN <= abs(total) <= MAX_TOTAL_TURN:
            return None
        sign = math.copysign(1.0, total)
        if any(turn * sign < -TURN_EPSILON for turn in turns):
            return None

        widest = max(widths)
        needed = SCORE_CAP * widest
        extent_a = self._plane_extent(main_a, rails[0].verts[0].co, axis, needed)
        extent_b = self._plane_extent(main_b, rails[-1].verts[0].co, axis, needed)
        score = min(extent_a, extent_b) / widest
        return _Run(faces, rails, main_a, main_b, widths, score, mode)

    def _plane_extent(self, face, origin, axis, needed):
        """How far the main plane reaches away from the bevel. Walks coplanar
        neighbours too, so triangulated n-gons still count as one wide face."""
        extent = max(_distance_to_line(v.co, origin, axis) for v in face.verts)
        if extent >= needed:
            return extent
        tolerance = needed * 1e-3
        visited = {face}
        queue = deque([face])
        while queue and len(visited) < MAX_PLANE_FACES:
            current = queue.popleft()
            for edge in current.edges:
                for neighbour in edge.link_faces:
                    if neighbour in visited or not _coplanar(face, neighbour, tolerance):
                        continue
                    visited.add(neighbour)
                    queue.append(neighbour)
                    extent = max(extent, max(_distance_to_line(v.co, origin, axis) for v in neighbour.verts))
                    if extent >= needed:
                        return extent
        return extent

    def _find_runs(self, mode):
        seeds = []
        for face in self.candidates:
            if face in self.taken or face in self.patch_faces:
                continue
            for orientation in (0, 1):
                width = self._quad_width(face, orientation)
                if width > EPSILON:
                    seeds.append((width, face.index, orientation, face))
        seeds.sort(key=lambda item: item[:3])  # narrowest first, deterministic

        runs = []
        covered = set()
        for _width, _index, orientation, face in seeds:
            if (face, orientation) in covered:
                continue
            section = self._grow(face, orientation, mode)
            if section is None:
                continue
            for member, rail in zip(section[0], section[1]):
                covered.add((member, self.quad_edges(member).index(rail) % 2))
            run = self._to_run(section, mode)
            if run is not None:
                runs.append(run)
        runs.sort(key=lambda item: item.score, reverse=True)
        return runs

    # -- Acceptance ----------------------------------------------------------

    def _accept_tier(self, runs, min_score):
        """Greedy by score; a run that uses a strip face as its main face is
        itself dropped (e.g. the small quad capping a pyramid apex)."""
        blocked = self.patch_faces
        rejected = set()
        while True:
            taken = {}
            chosen = []
            for run in runs:
                if run.score < min_score:
                    break
                if id(run) in rejected or run.faces[0] in self.taken:
                    continue
                if any(f in self.taken or f in taken or f in blocked or f in self.main_faces for f in run.faces):
                    continue
                if any(f in self.taken or f in blocked for f in (run.main_a, run.main_b)):
                    continue
                chosen.append(run)
                for face, rail in zip(run.faces, run.rails):
                    taken[face] = self.quad_edges(face).index(rail) % 2
            conflicts = [run for run in chosen if run.main_a in taken or run.main_b in taken]
            if not conflicts:
                break
            rejected.update(id(run) for run in conflicts)

        for run in chosen:
            self.accepted.append(run)
            self.main_faces.update((run.main_a, run.main_b))
            for face in run.faces:
                self.taken[face] = taken[face]
                self.run_of[face] = run
        self._update_patches()

    def detect(self):
        """Accept confident strips first: their corner patches then shield
        the patch faces from being mistaken for strips in later tiers."""
        runs = self._find_runs(self.mode)
        for min_score in SCORE_TIERS:
            self._accept_tier(runs, min_score)
        if self.mode == 'MANUAL':
            self._accept_tier(runs, 0.0)
        else:
            self._accept_tier(runs, MAIN_FACE_RATIO)
            # Same bevel width elsewhere: resolves chamfers nearly as wide as the faces around them.
            self.learned_widths = self._learn_widths()
            if self.learned_widths:
                self._accept_tier(self._find_runs('LEARNED'), LEARNED_MAIN_RATIO)
        self._drop_broken_strips()

    def _learn_widths(self):
        widths = []
        for run in self.accepted:
            for width in run.widths:
                if not any(abs(width - known) <= LEARNED_WIDTH_TOLERANCE * known for known in widths):
                    widths.append(width)
        return widths

    # -- Strip consistency ---------------------------------------------------

    def _continues_run(self, face, edge, run):
        """True if a rejected quad is really the same strip further along."""
        if not self._absorbable(face):
            return False
        orientation = (self.quad_edges(face).index(edge) + 1) % 2
        section = self._grow(face, orientation, run.mode)
        if section is None or None in section[2]:
            return False
        tolerance = max(run.widths) * 0.25
        main_a, main_b = section[2]
        return (
            (_coplanar(main_a, run.main_a, tolerance) and _coplanar(main_b, run.main_b, tolerance))
            or (_coplanar(main_a, run.main_b, tolerance) and _coplanar(main_b, run.main_a, tolerance))
        )

    def _drop_broken_strips(self):
        """Reject whole strips that cannot collapse without tearing the mesh."""
        strip_of = {}
        strips = []
        for face in self.taken:
            if face in strip_of:
                continue
            strip = []
            strip_of[face] = len(strips)
            queue = deque([face])
            while queue:
                current = queue.popleft()
                strip.append(current)
                for edge in self.cross_edges(current, self.taken[current]):
                    other = _other_face(edge, current)
                    if other in self.taken and other not in strip_of and edge in self.cross_edges(other, self.taken[other]):
                        strip_of[other] = len(strips)
                        queue.append(other)
            strips.append(strip)

        broken = set()
        for face, orientation in self.taken.items():
            for edge in self.cross_edges(face, orientation):
                other = _other_face(edge, face)
                if other is None:
                    continue
                if other in self.taken:
                    if edge not in self.cross_edges(other, self.taken[other]):
                        broken.add(strip_of[face])
                elif self._continues_run(other, edge, self.run_of[face]):
                    broken.add(strip_of[face])

        if not broken:
            return
        dropped = {face for index in broken for face in strips[index]}
        self.skipped += len(broken)
        self.accepted = [run for run in self.accepted if run.faces[0] not in dropped]
        self.taken = {face: value for face, value in self.taken.items() if face not in dropped}
        self._update_patches()

    # -- Corner patches ------------------------------------------------------

    def _update_patches(self):
        self.patches = self._find_patches()
        self.patch_faces = {face for patch in self.patches for face in patch}

    def _find_patches(self):
        """Faces enclosed exclusively by strip ends (bevel corners)."""
        strip_edges = set()
        cross = set()
        for face, orientation in self.taken.items():
            strip_edges.update(self.quad_edges(face))
            cross.update(self.cross_edges(face, orientation))

        patches = []
        visited = set()
        for edge in cross:
            for start in edge.link_faces:
                if start in self.taken or start in visited:
                    continue
                component = []
                valid = True
                visited.add(start)
                queue = deque([start])
                while queue:
                    face = queue.popleft()
                    component.append(face)
                    if len(component) > MAX_PATCH_FACES:
                        valid = False
                    for face_edge in face.edges:
                        neighbours = [f for f in face_edge.link_faces if f != face]
                        if face_edge in strip_edges:
                            if face_edge not in cross or any(f not in self.taken for f in neighbours):
                                valid = False
                            continue
                        if not neighbours:
                            valid = False
                        for neighbour in neighbours:
                            if neighbour not in visited:
                                visited.add(neighbour)
                                queue.append(neighbour)
                if valid:
                    patches.append(component)
        return patches

    def vertex_groups(self):
        """Vertices that merge into one original vertex each."""
        union = _UnionFind()
        for face, orientation in self.taken.items():
            for edge in self.cross_edges(face, orientation):
                union.union(edge.verts[0], edge.verts[1])
        for patch in self.patches:
            verts = [v for face in patch for v in face.verts]
            for vert in verts[1:]:
                union.union(verts[0], vert)

        groups = {}
        for vert in list(union.parent):
            groups.setdefault(union.find(vert), []).append(vert)
        return [group for group in groups.values() if len(group) > 1]


def _solve_corner(group, removed_faces, removed_edges):
    """Least-squares point on all main planes touching the group."""
    group_set = set(group)
    points = np.array([v.co for v in group], dtype=np.float64)
    centroid = points.mean(axis=0)

    planes = []
    seen = set()
    for vert in group:
        for face in vert.link_faces:
            if face in removed_faces or face in seen:
                continue
            seen.add(face)
            normal = np.array(face.normal, dtype=np.float64)
            if np.dot(normal, normal) < 0.5:
                continue
            planes.append((normal, float(np.dot(normal, np.array(face.calc_center_median())))))
    if not planes:
        return centroid, 0

    normals = np.array([plane[0] for plane in planes])
    offsets = np.array([plane[1] for plane in planes])
    residual = offsets - normals @ centroid
    left, singular, right = np.linalg.svd(normals)
    rank = int(np.sum(singular > singular[0] * PLANE_RANK_RATIO))
    point = centroid.copy()
    for index in range(rank):
        point += (left[:, index] @ residual / singular[index]) * right[index]

    if rank == 2:
        point += right[2] * _slide_along_crease(group, group_set, point, right[2], removed_edges)
    return point, rank


def _slide_along_crease(group, group_set, origin, axis, removed_edges):
    """Where the kept edges running into the bevel meet the restored crease."""
    shifts = []
    for vert in group:
        for edge in vert.link_edges:
            if edge in removed_edges:
                continue
            other = edge.other_vert(vert)
            if other in group_set:
                continue
            start = np.array(other.co, dtype=np.float64)
            direction = np.array(vert.co, dtype=np.float64) - start
            length = np.linalg.norm(direction)
            if length < EPSILON:
                continue
            direction /= length
            alignment = float(np.dot(axis, direction))
            denominator = 1.0 - alignment * alignment
            if denominator < 0.03:  # nearly parallel to the crease
                continue
            offset = origin - start
            shifts.append((alignment * np.dot(direction, offset) - np.dot(axis, offset)) / denominator)
    return float(np.mean(shifts)) if shifts else 0.0


def _fit_uv_maps(bm, faces, uv_layers):
    """Affine plane->UV map per face and layer, fitted on the current shape."""
    maps = []
    for face in faces:
        normal = face.normal
        if normal.length < 0.5 or len(face.loops) < 3:
            continue
        tangent = normal.orthogonal().normalized()
        bitangent = normal.cross(tangent)
        coords = np.array([(loop.vert.co.dot(tangent), loop.vert.co.dot(bitangent), 1.0) for loop in face.loops])
        if np.linalg.matrix_rank(coords, tol=1e-9) < 3:
            continue
        for layer in uv_layers:
            uvs = np.array([tuple(loop[layer].uv) for loop in face.loops])
            matrix = np.linalg.lstsq(coords, uvs, rcond=None)[0]
            maps.append((face, layer, tangent, bitangent, matrix))
    return maps


def unbevel_bmesh(bm, max_width=0.0, mark_sharp=True, correct_uvs=True):
    """Collapse bevel strips in the selected faces of bm. Returns stats dict."""
    bm.normal_update()
    bm.faces.index_update()
    detector = _BevelDetector(bm, max_width)
    detector.detect()
    stats = {"strips": 0, "corners": 0, "skipped": detector.skipped, "approximated": 0}
    if not detector.taken:
        return stats

    removed_faces = set(detector.taken) | detector.patch_faces
    removed_edges = {edge for face in removed_faces for edge in face.edges}
    groups = detector.vertex_groups()

    targets = {}
    for group in groups:
        point, rank = _solve_corner(group, removed_faces, removed_edges)
        if rank < 2:
            stats["approximated"] += 1
        targets[id(group)] = point
        if rank >= 3:
            stats["corners"] += 1

    uv_maps = []
    if correct_uvs and bm.loops.layers.uv:
        moved = {v for group in groups for v in group}
        touched = {f for v in moved for f in v.link_faces if f not in removed_faces}
        uv_maps = _fit_uv_maps(bm, touched, list(bm.loops.layers.uv.values()))

    representative = {}
    targetmap = {}
    for group in groups:
        point = targets[id(group)]
        keep = group[0]
        for vert in group:
            vert.co = point
            representative[vert] = keep
            if vert is not keep:
                targetmap[vert] = keep

    for face, layer, tangent, bitangent, matrix in uv_maps:
        for loop in face.loops:
            if loop.vert in representative:
                co = loop.vert.co
                loop[layer].uv = tuple(np.array((co.dot(tangent), co.dot(bitangent), 1.0)) @ matrix)

    creases = []
    for run in detector.accepted:
        verts = run.rails[0].verts
        creases.append((representative.get(verts[0], verts[0]), representative.get(verts[1], verts[1])))

    bmesh.ops.weld_verts(bm, targetmap=targetmap)

    if mark_sharp:
        for first, second in creases:
            if first.is_valid and second.is_valid and first != second:
                edge = bm.edges.get((first, second))
                if edge is not None:
                    edge.smooth = False

    stats["strips"] = len({frozenset(pair) for pair in creases if pair[0] != pair[1]})
    bm.normal_update()
    return stats


# --- OPERATORS ---

class PM_OT_Unbevel(bpy.types.Operator):
    """Collapse bevels/chamfers baked into the selected faces back into sharp edges.
Select all (A) in Edit Mode and run"""
    bl_idname = "pm.unbevel"
    bl_label = "Unbevel"
    bl_options = {'REGISTER', 'UNDO'}

    max_width: bpy.props.FloatProperty(  # type: ignore
        name="Max Width",
        description="0 = detect bevels automatically. Otherwise every strip narrower than this is treated as a bevel",
        default=0.0,
        min=0.0,
        subtype='DISTANCE',
        unit='LENGTH',
        precision=4,
        step=0.1,
    )
    mark_sharp: bpy.props.BoolProperty(  # type: ignore
        name="Mark Sharp",
        description="Mark the restored edges as sharp so smooth shading keeps them crisp",
        default=True,
    )
    correct_uvs: bpy.props.BoolProperty(  # type: ignore
        name="Correct UVs",
        description="Extend the UVs of the remaining faces instead of stretching them",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        totals = {"strips": 0, "corners": 0, "skipped": 0, "approximated": 0}
        for obj in context.objects_in_mode_unique_data:
            if obj.type != 'MESH':
                continue
            bm = bmesh.from_edit_mesh(obj.data)
            stats = unbevel_bmesh(bm, self.max_width, self.mark_sharp, self.correct_uvs)
            if stats["strips"]:
                bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
            for key, value in stats.items():
                totals[key] += value

        if not totals["strips"]:
            self.report({'WARNING'}, "No bevels found in the selected faces")
            return {'CANCELLED'}
        message = f"Unbevel: {totals['strips']} edges restored"
        if totals["skipped"]:
            message += f", {totals['skipped']} ambiguous strips skipped"
        self.report({'INFO'}, message)
        return {'FINISHED'}


# --- UI DRAWING ---

def draw_ui(layout, context):
    box = layout.box()
    box.label(text="Mesh Cleanup", icon='MOD_BEVEL')
    box.operator("pm.unbevel", text="Unbevel", icon='MOD_BEVEL')


# --- REGISTRATION ---

classes = (
    PM_OT_Unbevel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
