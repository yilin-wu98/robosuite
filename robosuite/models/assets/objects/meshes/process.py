import open3d as o3d
import os
import numpy as np
mesh = o3d.io.read_triangle_mesh("cup.obj", True)
        # mesh = o3d.io.read_triangle_mesh(obj_path, True)
mesh.compute_vertex_normals()
        # pcd = o3d.geometry.sample_points_uniformly(mesh, number_of_points=10000)
        # points = np.asarray(pcd.points)
        # pcd_scaled = o3d.geometry.PointCloud()
        # pcd_scaled.points = o3d.utility.Vector3dVector(points)
vertices = np.asarray(mesh.vertices)
print(vertices.shape)
print("Water tight? ", mesh.is_watertight())

max_points = np.max(vertices, axis=0)
min_points = np.min(vertices, axis=0)
print(max_points)
print(min_points)
size = (max_points-min_points)/2
origin = (max_points+min_points)/2
        
stl_file = 'cup' + '.stl'
stl_path = os.path.join('./', stl_file)
        # o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh(stl_path, mesh)
