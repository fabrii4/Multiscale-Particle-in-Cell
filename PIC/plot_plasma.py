import os
import numpy as np
import trimesh
from trimesh import viewer

N_particles=100000
N_steps=20000
n_skip=2

#read parameters from file
with open("param.txt", "r") as file:
    for line in file:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("Np"):
            elements = line.split()
            N_particles=int(elements[1])
        if line.startswith("Nsave"):
            elements = line.split()
            N_steps=int(elements[1])
            break

#current loop rings dimensions (as defined in Poisson.cu, should be moved to param.txt)
EXT_AR=0.01
EXT_DR=0.01

color_pos = [0,255,0,255]
color_neg = [255,0,0,255]

import struct

# Definition of your data format: 
# '<3f' means Little-Endian (<) and three 32-bit floats (3f)
format_string = '<4f'
chunk_size = N_particles*struct.calcsize(format_string)  # Matches 12 bytes total
path='./results.bin'

binary_file = open(path, 'rb')

def get_points_from_file():
    global binary_file, n_skip
    raw_bytes = binary_file.read(chunk_size)
    #skip n_skip frames
    binary_file.seek(n_skip*chunk_size, os.SEEK_CUR)
    if len(raw_bytes) < chunk_size:
            binary_file.seek(0)
            raw_bytes = binary_file.read(chunk_size)
    data = np.frombuffer(raw_bytes, dtype=np.float32)
    data = data.reshape(N_particles, 4)
    points = data[:,:3].astype(np.float16)
    charges = np.expand_dims(data[:,3],axis=1)
    colors = np.where(charges < 0, color_neg, color_pos)
    return points, colors
                
n=1
def update_point_cloud(scene):
    points, colors = get_points_from_file()
    pcl = trimesh.PointCloud(vertices=points, colors=colors)
    scene.geometry['points'] = pcl

#points=traj[0]
#colors=traj_color[0]
points, colors = get_points_from_file()
pcl = trimesh.PointCloud(vertices=points, colors=colors)
scene=trimesh.Scene()

#rotate camera
current_transform = scene.camera_transform
angle = np.radians(90)
rotation = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
translation = trimesh.transformations.translation_matrix([0, 0, 0.09])
new_transform = current_transform @ rotation @ translation
scene.camera_transform = new_transform

grid_geometry = trimesh.path.creation.grid(side=5*EXT_AR, count=10, include_circle=False)
scene.add_geometry(grid_geometry, transform=trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0]))
torus = trimesh.creation.torus(major_radius=EXT_AR, minor_radius=0.2*EXT_AR)
scene.add_geometry(torus, transform=trimesh.transformations.translation_matrix([0.0, 0.0, EXT_DR]))
scene.add_geometry(torus, transform=trimesh.transformations.translation_matrix([0.0, 0.0, -EXT_DR]))
scene.add_geometry(pcl, geom_name='points')

#display scene
viewer.SceneViewer(scene, resolution=(800,600), line_settings={'point_size':1, 'line_width':1},
                   background=[30,30,60,255], callback=update_point_cloud, callback_period=0.001)
