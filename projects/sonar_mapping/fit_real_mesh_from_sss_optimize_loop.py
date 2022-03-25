import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene, Pointclouds
from pytorch3d.renderer.materials import Materials

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRendererWithFragments4SSS, softmax_rgb_blend, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader,SoftPhongShader4SSS, PointLights, DirectionalLights, TexturesVertex, Textures,
)

# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
    point_mesh_face_distance
)
from pytorch3d.utils import ico_sphere
import sys 
import math 

from math import radians, degrees
import json
import cv2 
import random
from scipy.ndimage.filters import gaussian_filter

def eulerAnglesToRotationMatrix(theta) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R, R_x, R_y, R_z



random.seed(0)

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
DATA_DIR = "./data"

# obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")
res = 0.5
# V = np.load("/media/yipingx/SSD/Data/gothernburg/gothernburg/cereal_file/V_res_0.5m.npy")
# F = np.load("/media/yipingx/SSD/Data/gothernburg/gothernburg/cereal_file/F_res_0.5m.npy")

heightmap_gt, bounds = np.load(DATA_DIR + "height_map_from_dtm.npz", encoding="latin1",allow_pickle=True)["arr_0"]

# heightmap = -np.ones_like(heightmap_gt) * 20.0
heightmap = gaussian_filter(heightmap_gt, sigma=10)
# heightmap[heightmap_gt==0]=0
# heightmap = heightmap_gt.copy()
# V = np.load("/media/yipingx/SSD/Data/gothernburg/gothernburg/cereal_file/V_res_10m.npy")
# F = np.load("/media/yipingx/SSD/Data/gothernburg/gothernburg/cereal_file/F_res_10m.npy")
V,F=mesh_map.mesh_from_height_map(heightmap, bounds)
# mesh_map.show_mesh(V,F)
verts = torch.from_numpy(V).float()
faces = torch.from_numpy(F).long()

# faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = 0.3*torch.ones_like(verts)[None]  # (1, V, 3)
# verts_rgb = torch.zeros_like(verts)[None]  # (1, V, 3)

textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
seafloor_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)

 
sensor_offset = np.array([2., -1.5, 0.],dtype=np.float32).reshape(-1,3)
sensor_yaw = 5.*math.pi/180.

tilt_angle = 30.0
beam_width = 50.0
horizontal_beam_width = 1.2
aspect_ratio = math.tan(math.radians(horizontal_beam_width/2)) / math.tan(math.radians(beam_width/2))

H_select = 1024
loop_select = 1024
loop_interval = 1
nbr_time_bins = 256
max_slant_range = 50.0

# gamma = 1e-4 # kernel = torch.exp(-z_diff**2 / blend_params.gamma)
gamma = 1e-6

sigma = 1e-1 # prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask 
# sigma = 1e-3 # prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask 

# blur_radius = 1e-5 #1e-4 #1e-8 

blur_radius = np.log(1. / 1e-4 - 1.) * sigma
# blur_radius = 0.1

print("sgima: ", sigma, "blur_radius: ", blur_radius)
# sigma (float): Controls the width of the sigmoid function used to
#             calculate the 2D distance based probability. Determines the
#             sharpness of the edges of the shape.
#             Higher => faces have less defined edges.
# gamma (float): Controls the scaling of the exponential function used
#             to set the opacity of the color.
#             Higher => faces are more transparent.


faces_per_pixel = 3

blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=(0.0, 0.0, 0.0))


up = (0, 0, 1) # ENU vector specifying the up direction in the world coordinate frame.

ups = [up]*H_select


# Optimize using rendered silhouette image loss, mesh edge loss, mesh normal 
# consistency, and mesh laplacian smoothing
losses = {"sss": {"weight": 1000.0, "values": []},
          "edge": {"weight": 1.0, "values": []},
          "normal": {"weight": 1.0, "values": []},
          "laplacian": {"weight": 0.01, "values": []},
          "sparse_depth": {"weight": 0.000000001, "values": []}, # 1000?

         }
print("losses weight ", losses )
image_size_ = 3

RESULTS_DIR = "./results-real-data" + "-H_select-"+ str(H_select) + "-image_size_-" + str(image_size_) +"-faces_per_pixel"+str(faces_per_pixel) +"-gamma"+str(gamma)+"-sigma"+str(sigma)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
np.save(RESULTS_DIR+os.sep+"initial_losses.npy", losses)

# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)
    
    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)
    
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")



image_size_ = 3
bin_size=36
# SAVE_DIR = "./real-data" + "-H_select-"+ str(H_select) + "-image_size_-" + str(image_size_) +"-faces_per_pixel"+str(faces_per_pixel) +"-gamma"+str(gamma)+"-sigma"+str(sigma)
SAVE_DIR = DATA_DIR

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

raster_settings = RasterizationSettings(
    # image_size=image_size, 
    # image_size=(image_size,math.ceil(image_size*aspect_ratio)),
    image_size=(math.floor(image_size_/aspect_ratio), image_size_),
    # image_size=(133,133),

    # perspective_correct=False,
    # image_size=(image_size,image_size//8), 
    # image_size=(int(image_size/aspect_ratio),image_size), 
    # blur_radius=np.log(1. / blur_radius- 1.) * blend_params.sigma, 
    blur_radius=blur_radius, 
    bin_size=bin_size,
    faces_per_pixel=faces_per_pixel, 
    # z_clip_value=5,
    # cull_backfaces=True,
    # cull_to_frustum=True,
    )
materials = Materials(ambient_color=((0, 0, 0),), diffuse_color=((1, 1, 1),)).to(device=device)
fx = 1/math.tan(math.radians(horizontal_beam_width/2))
# fy = 1/math.tan(math.radians(beam_width/2))
fy=fx
cameras = PerspectiveCameras(
    focal_length=((fx, fy),),
    principal_point=((0.0, 0.0),),
    # R=R,
    # T=T,
    # K: Optional[torch.Tensor] = None,
    device=device)
phong_renderer4sss = MeshRendererWithFragments4SSS(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader4SSS(device=device, cameras=cameras, materials=materials,blend_params=blend_params )
)
meshes = seafloor_mesh.extend(1)

# verts_shape = meshes.verts_packed().shape
verts_shape = meshes.verts_packed()[:,1:2].shape
deform_xy_verts = torch.full(meshes.verts_packed()[:,:2].shape, 0.0, device=device, requires_grad=False)

deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
# print("deform_verts.shape ", deform_verts.shape) # deform_verts.shape  torch.Size([2560000, 3])
# The optimizer
# optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
optimizer = torch.optim.Adam([deform_verts], lr=0.2)
Niter = 50
loop = tqdm(range(Niter))

plot_period = 1
lines = {15:600,18:400, 26:1000, 64:500} # len: 1686, 2509, 2160, 1583

clip = 1e5
batch_size = 32
with torch.autograd.set_detect_anomaly(True):  
    for i in loop:
        for line, start in lines.items():
            print("line ", line, " start ", start)
            # Initialize optimizer

            sss_rendered_target =  torch.from_numpy(np.load(SAVE_DIR+os.sep+"sss_rendered-line"+str(line)+"-start"+str(start)+".npy")).float().to(device)
        
            camera_positions = torch.from_numpy(np.load(SAVE_DIR+os.sep+"camerea_positions-line"+str(line)+"-start"+str(start)+".npy")).float()
            ats = torch.from_numpy(np.load(SAVE_DIR+os.sep+"ats_positions-line"+str(line)+"-start"+str(start)+".npy")).float()   
            altitudes = np.load(SAVE_DIR + os.sep+"altitude-line"+str(line)+"-start"+str(start)+".npy")



            camera_positions = camera_positions.to(device=device)
            R = look_at_rotation(camera_positions, 
                device=device, 
                at=ats, 
                up= ups) # (256, 3, 3)
            
            
            T = -torch.bmm(R.transpose(1, 2), camera_positions.unsqueeze(2))[:, :, 0]   # (1, 3)

            loss = {k: torch.tensor(0.0, device=device) for k in losses}
            deform_verts_full = torch.cat((deform_xy_verts, deform_verts), dim=1)
            new_src_mesh = meshes.to(device).offset_verts(deform_verts_full)

            sss_rendered_ref_all = np.zeros((H_select,nbr_time_bins))
            for loop_ in range(loop_select):   
                if loop_%batch_size==0 and loop_!=0:
                    optimizer.zero_grad()
                    # Deform the mesh
                    deform_verts_full = torch.cat((deform_xy_verts, deform_verts), dim=1)
                    new_src_mesh = meshes.to(device).offset_verts(deform_verts_full)

                    # calculate regularization related loss
                    update_mesh_shape_prior_losses(new_src_mesh, loss)

                    # Weighted sum of the losses
                    sum_loss = torch.tensor(0.0, device=device)
                    if i>Niter//2:
                        losses["sparse_depth"]["weights"]=0.0005
                    for k, l in loss.items():
                        sum_loss += l * losses[k]["weight"]
                        losses[k]["values"].append(float(l.detach().cpu()))

                    if loop_%320==0 :
                    
                        # Print the losses
                        loop.set_description("total_loss = %.6f" % sum_loss)
                        for k, l in losses.items():
                            print(str(k)+" : "+str(l["values"][-1]*l["weight"]))
                    

                        # print("V max", np.max(new_V, axis=0))
                        # print("V min", np.min(new_V, axis=0))

                        # mesh_map.show_mesh(new_V, new_F)
                    # Optimization step
                    sum_loss.backward()
                    torch.nn.utils.clip_grad_norm_([deform_verts], clip)
                    optimizer.step() # deform_verts is changed here
                    loss = {k: torch.tensor(0.0, device=device) for k in losses}



                index_start = loop_ * loop_interval
                index_stop = loop_ * loop_interval  + loop_interval
                lights = PointLights(device=device, location=camera_positions[index_start:index_stop],
                    ambient_color=((0.0, 0.0, 0.0),),
                    diffuse_color=((1.0, 1.0, 1.0),),
                    specular_color=((0.0, 0.0, 0.0),),)
                sss_rendered, fragments = phong_renderer4sss(meshes_world=new_src_mesh, R=R[index_start:index_stop], T=T[index_start:index_stop], lights=lights, eps=1e-10)

                # sss_rendered =  softmax_rgb_blend_4sss_p(colors, fragments, blend_params)

                # sss_rendered_ref = sss_rendered
                # sss_rendered_ref_ = sss_rendered_ref
                sss_rendered_ref_all[loop_] = sss_rendered.detach().cpu().numpy()
                loss_silhouette = ((sss_rendered[0] - sss_rendered_target[index_start:index_stop]) ** 2).mean()
                
                loss["sss"] += loss_silhouette 
                points_under_auv = Pointclouds([ camera_positions[loop_].reshape(-1,3) + torch.tensor([[0., 0., -altitudes[loop_] ]]).float().to(device) ])
                loss["sparse_depth"] += point_mesh_face_distance_4SSS(new_src_mesh, points_under_auv) # squared distance in meter


        # # Plot mesh
        if i % plot_period == 0:
            new_V = new_src_mesh.verts_packed().detach().cpu().numpy()
            new_F = new_src_mesh.faces_packed().detach().cpu().numpy()
            # mesh_map.show_mesh(new_V, new_F)
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 1, 1)
            plt.imshow(sss_rendered_ref_all, cmap="gray")
            plt.title("sss_rendered_ref_line "+str(line))
            np.save(RESULTS_DIR+os.sep+"sss_rendered_ref_line "+str(line)+"at-epoch-"+str(i)+".npy", sss_rendered_ref_all)
            plt.savefig(RESULTS_DIR+os.sep+"sss_rendered_ref_line"+str(line)+"-epoch"+str(i)+ ".png")

            plt.subplot(2, 1, 2)
            plt.imshow(sss_rendered_target.cpu().numpy(), cmap="gray")
            plt.title("sss_rendered_target line "+str(line))
            np.save(RESULTS_DIR+os.sep+"sss_rendered_target_line "+str(line)+"at-epoch-"+str(i)+".npy", sss_rendered_target.cpu().numpy())
            plt.savefig(RESULTS_DIR+os.sep+"sss_rendered_target_line"+str(line)+"-epoch"+str(i)+ ".png")


            # save mesh
            new_V = new_src_mesh.verts_packed().detach().cpu().numpy()
            new_F = new_src_mesh.faces_packed().detach().cpu().numpy()

            np.save(RESULTS_DIR+os.sep+"v-at-epoch-"+str(i)+".npy", new_V)
            np.save(RESULTS_DIR+os.sep+"f-at-epoch-"+str(i)+".npy", new_F)

np.save(RESULTS_DIR+os.sep+"losses.npy", losses)



