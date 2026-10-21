"""Curve to Delaunay: fill a closed flat curve with an even triangle mesh.

Geometry Nodes has no Delaunay node, but Fill Curve (Triangles) is a
constrained Delaunay triangulation that also keeps single-point curves as
vertices. So every boundary segment is split into even pieces (the original
points, and with them sharp corners, stay), Poisson points are scattered inside,
each point becomes its own curve, and Fill Curve triangulates everything.
"""

import bpy

UI_CATEGORY = "SCENE_OPTIMIZATION"

NODE_GROUP_NAME = "PM_CurveToDelaunay"
MODIFIER_NAME = "Delaunay Fill"
# Bump when the node tree changes so groups saved in .blend files get rebuilt.
NODE_GROUP_VERSION = 3


# --- LOGIC ---

def _build_node_group(ng=None):
    if ng is None:
        ng = bpy.data.node_groups.new(NODE_GROUP_NAME, 'GeometryNodeTree')
    # Keep an existing interface so modifier values survive the rebuild.
    if not ng.interface.items_tree:
        ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        size = ng.interface.new_socket("Edge Length", in_out='INPUT', socket_type='NodeSocketFloat')
        size.default_value, size.min_value, size.subtype = 0.2, 0.01, 'DISTANCE'
        ng.interface.new_socket("Seed", in_out='INPUT', socket_type='NodeSocketInt')
        ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    ng.nodes.clear()
    ng["pm_version"] = NODE_GROUP_VERSION

    nodes, link = ng.nodes, ng.links.new
    group_in = nodes.new('NodeGroupInput')
    group_out = nodes.new('NodeGroupOutput')
    curve, edge, seed = group_in.outputs[0], group_in.outputs[1], group_in.outputs[2]

    def math(operation, a, b):
        node = nodes.new('ShaderNodeMath')
        node.operation = operation
        for socket, value in zip(node.inputs, (a, b)):
            if isinstance(value, bpy.types.NodeSocket):
                link(value, socket)
            else:
                socket.default_value = value
        return node.outputs[0]

    # Cuts per segment from the straight distance to the next point, so the
    # original points stay. shortcut: chord, not arc length, so curvy Bezier
    # segments get a bit fewer points; NURBS control points are off the curve.
    next_point = nodes.new('GeometryNodeOffsetPointInCurve')
    next_point.inputs['Offset'].default_value = 1
    next_position = nodes.new('GeometryNodeFieldAtIndex')
    next_position.data_type, next_position.domain = 'FLOAT_VECTOR', 'POINT'
    position = nodes.new('GeometryNodeInputPosition').outputs[0]
    link(next_point.outputs['Point Index'], next_position.inputs['Index'])
    link(position, next_position.inputs['Value'])
    distance = nodes.new('ShaderNodeVectorMath')
    distance.operation = 'DISTANCE'
    link(position, distance.inputs[0])
    link(next_position.outputs[0], distance.inputs[1])
    subdivide = nodes.new('GeometryNodeSubdivideCurve')
    link(curve, subdivide.inputs['Curve'])
    link(math('SUBTRACT', math('CEIL', math('DIVIDE', distance.outputs['Value'], edge), 0.0), 1.0),
         subdivide.inputs['Cuts'])
    # Resolution 1: the boundary is exactly the control points.
    boundary = nodes.new('GeometryNodeSetSplineResolution')
    boundary.inputs['Resolution'].default_value = 1
    link(subdivide.outputs[0], boundary.inputs['Geometry'])

    area = nodes.new('GeometryNodeFillCurve')
    area.inputs['Mode'].default_value = 'N-gons'
    link(boundary.outputs[0], area.inputs['Curve'])

    scatter = nodes.new('GeometryNodeDistributePointsOnFaces')
    scatter.distribute_method = 'POISSON'
    link(area.outputs[0], scatter.inputs['Mesh'])
    link(edge, scatter.inputs['Distance Min'])
    # Poisson needs many more candidates than the ~1.15/d^2 points that survive.
    link(math('DIVIDE', 10.0, math('MULTIPLY', edge, edge)), scatter.inputs['Density Max'])
    link(seed, scatter.inputs['Seed'])

    # Points too close to the boundary would make slivers.
    rails = nodes.new('GeometryNodeCurveToMesh')
    link(boundary.outputs[0], rails.inputs['Curve'])
    proximity = nodes.new('GeometryNodeProximity')
    proximity.target_element = 'EDGES'
    link(rails.outputs[0], proximity.inputs['Target'])
    trim = nodes.new('GeometryNodeDeleteGeometry')
    trim.domain = 'POINT'
    link(scatter.outputs['Points'], trim.inputs['Geometry'])
    link(math('LESS_THAN', proximity.outputs['Distance'], math('MULTIPLY', edge, 0.7)), trim.inputs['Selection'])

    # One curve per point, so Fill Curve keeps them as vertices.
    singles = nodes.new('GeometryNodePointsToCurves')
    link(trim.outputs[0], singles.inputs['Points'])
    link(nodes.new('GeometryNodeInputIndex').outputs[0], singles.inputs['Curve Group ID'])

    join = nodes.new('GeometryNodeJoinGeometry')
    link(singles.outputs[0], join.inputs[0])
    link(boundary.outputs[0], join.inputs[0])
    fill = nodes.new('GeometryNodeFillCurve')
    fill.inputs['Mode'].default_value = 'Triangles'
    link(join.outputs[0], fill.inputs['Curve'])
    link(fill.outputs[0], group_out.inputs[0])

    # Columns by link depth, so the tree is readable when opened.
    depth = {node: 0 for node in nodes}
    for _ in nodes:
        for node_link in ng.links:
            depth[node_link.to_node] = max(depth[node_link.to_node], depth[node_link.from_node] + 1)
    rows = {}
    for node in nodes:
        row = rows[depth[node]] = rows.get(depth[node], -1) + 1
        node.location = (depth[node] * 220.0, -row * 220.0)
    return ng


def get_node_group():
    ng = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if ng is not None and ng.get("pm_version") == NODE_GROUP_VERSION:
        return ng
    return _build_node_group(ng)


# --- OPERATORS ---

class PM_OT_CurveToDelaunay(bpy.types.Operator):
    """Fill the selected closed curves with an even triangle mesh (Geometry Nodes modifier).
The curve must be flat in its local XY plane"""
    bl_idname = "pm.curve_to_delaunay"
    bl_label = "Curve to Delaunay"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(obj.type == 'CURVE' for obj in context.selected_objects)

    def execute(self, context):
        node_group = get_node_group()
        count = 0
        for obj in context.selected_objects:
            if obj.type != 'CURVE' or obj.modifiers.get(MODIFIER_NAME):
                continue
            obj.modifiers.new(MODIFIER_NAME, 'NODES').node_group = node_group
            count += 1
        self.report({'INFO'}, f"Delaunay Fill added to {count} curves")
        return {'FINISHED'}


# --- UI DRAWING ---

def draw_ui(layout, context):
    box = layout.box()
    box.label(text="Curve Fill", icon='MOD_TRIANGULATE')
    box.operator("pm.curve_to_delaunay", text="Curve to Delaunay", icon='MOD_TRIANGULATE')


# --- REGISTRATION ---

classes = (
    PM_OT_CurveToDelaunay,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
