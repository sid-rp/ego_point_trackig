import os
from concurrent.futures import ThreadPoolExecutor
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import matplotlib.pyplot as plt
import pyrender
import torch
import scipy.stats as stats
from PIL import Image
from scipy.optimize import linear_sum_assignment
import cv2
import torchvision.transforms as T
from colordict import *
from trimesh import path
import os
import math
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

rescale_scores = {
    'P03_04': 0.9744356799,
    'P05_08': 0.6808696067,
    'P06_05': 0.5479392007,
    'P07_10': 0.7000619555,
    'P07_101': 0.8526057764,
    'P09_14': 0.807294715,
    'P11_106': 0.7854287263,
    'P12_04': 0.747035204,
    'P12_101': 0.4651162791,
    'P19_04': 0.5028916269,
    'P24_08': 3.95,
    'P14_05': 2.13684,
    'P20_04': None,
    'P21_01': None,
    'P22_114': 0.8423322495,
    'P23_04': 0.6576783953,
    'P23_112': 0.6657656638,
    'P24_05': 0.5391126206,
    'P25_05': 0.7407407407,
    'P26_124': 0.9245177484,
    'P28_103': 1.143961883,
    'P30_107': 0.8297412452,
    'P01_14': 1.265822785,
    'P02_09': 1.025499033,
    'P04_05': 0.6309944472,
    'P06_101': 0.7367947953,
    'P08_05': 0.6336934825,
    'P10_04': 0.7426248074,
    'P11_05': 0.904159132,
    'P13_08': 0.5752085131,
    'P14_08': 0.7861017216,
    'P15_08': 0.350262697,
    'P17_04': 0.4575528245,
    'P18_05': 0.4780114723,
    'P27_103': None,
    'P01_109': None,
    '22_15': None,
    'P30_05': 0.6166178511,
    'P32_06': 0.3060069158,
    'P11_101': 0.8367080558,
    'P04_12': None,
    'P35_105': 0.826446281,
    'P02_121': 0.8695652174,
    'P15_02': 0.5714285714,
    'P02_107': 0.9950248756,
    'P02_132': 1.342281879,
    'P32_01': 0.6097560976,
    'P22_01': 0.8240626288,
    'P06_11': 1.069518717,
    'P11_104': None,
    'P02_102': 1.388888889,
    'P35_109': 0.956937799,
    'P02_124': 0.8220304151,
    'P08_16': 0.5917159763,
    'P13_10': None,
    'P11_107': None,
    'P02_122': 1.27388535,
    'P37_101': 0.9550642281,
    'P28_110': None,
    'P20_03': 1.321003963,
    'P02_09': 1.020408163,
    'P04_121': 0.4246284501,
    'P37_103': 0.5221932115,
    'P02_101': 0.9615384615,
    'P01_03': 1.739130435,
    'P27_101': 0.9756097561,
    'P28_113': 1.538461538,
    'P01_103': 1.136363636,
    'P30_111': 0.9049773756,
    'P03_13': 0.78125,
    'P02_132': 1.360544218,
    'P23_02': 0.7462686567,
    'P28_14': 1.321615014,
    'P14_05': None,
    'P27_101': 0.9708737864,
    'P28_113': 1.324503311,
    'P01_103': 1.19047619,
    'P30_111': 1.209189843,
    'P03_13': 0.4842615012,
    'P02_132': 1.307189542,
    'P23_02': 0.7490636704,
    'P28_14': 0.9900990099,
    'P01_104': 0.826446281,
    'P01_01': 1.234567901,
    'P28_13': 0.3853564547,
    'P03_17': 0.4975124378,
    'P03_13': 0.7434944238,
    'P02_01': 0.9478672986,
    'P18_07': 0.7662835249,
    'P04_21': 0.6872852234,
    'P04_25': 0.6309148265,
    'P03_113': 0.5899705015,
    'P03_24': 0.7633587786,
    'P03_11': 0.6666666667,
    'P23_05': 0.8097165992,
    'P01_09': 1.19760479,
    'P10_04': 0.7604562738,
    'P04_26': 0.422832981,
    'P08_21': 0.701754386,
    'P11_16': 0.9049773756,
    'P13_08': 0.8230452675,
    'P03_23': 0.4938271605,
    'P18_07': 0.6993006993,
    'P02_124': 0.8403361345,
    'P11_104': 0.3710575139,
    'P03_03': 0.625,
    'P18_06': 0.4140786749,
    'P04_33': 0.6349206349,
    'P02_128': 1.081081081,
    'P13_10': None,
    'P07_08': 0.826446281,
    'P12_02': 1.069518717,
    'P18_03': 0.8163265306,
    'P03_101': 0.8333333333,
    'P37_101': 0.9174311927,
    'P28_06': 0.9259259259,
    'P25_107': 0.4545454545,
    'P28_110': None,
    'P03_112': 0.9433962264,
    'P28_14': 1.369863014,
    'P04_33': 0.7042253521,
    'P28_101': 1.156069364,
    'P26_110': 0.5617977528,
    'P22_117': 0.9302325581,
    'P28_112': 1.526717557,
    'P32_01': 0.5698005698,
    'P37_103': 0.4807692308,
    'P04_02': 0.9661835749,
    'P04_109': 0.5025125628,
    'P04_101': 0.6756756757,
    'P03_11': 0.6666666667,
    'P23_05': 0.8968609865,
    'P30_101': 0.5221932115,
    'P02_130': 1.03626943,
    'P08_16': 0.4424778761,
    'P06_110': 0.4210526316,
    'P06_13': 0.583090379,
    'P04_11': 0.5494505495,
    'P02_122': 1.27388535,
    'P11_101': 0.5319148936,
    'P04_110': 0.7547169811,
    'P08_09': 0.6622516556,
    'P06_01': 0.7299270073,
    'P06_07': 0.487804878,
    'P12_03': 0.7246376812,
    'P05_01': 0.6006006006,
    'P02_07': 1.324503311,
    'P06_107': 0.7874015748,
    'P06_102': 0.625,
    'P06_14': 0.8583690987,
    'P28_109': 1.360544218,
    'P02_109': 1.265822785,
    'P04_114': 0.4608294931,
    'P04_03': 0.9302325581,
    'P11_102': 1.149425287,
    'P03_05': 0.4618937644,
    'P11_105': 1.030927835,
    'P22_07': 0.826446281,
    'P03_123': 0.4535147392,
    'P06_103': 0.9900990099,
    'P30_112': 0.4184100418,
    'P01_05': 1.19760479,
    'P02_03': 1.219512195,
    'P06_09': 0.6269592476,
    'P06_12': 1.183431953,
    'P17_01': 1.098901099,
    'P02_135': 1.104972376,
    'P11_103': 0.7407407407,
    'P04_04': 0.9852216749

}


def is_segment_intersecting_mesh_1(segment_start, segment_end, mesh, threshold=0.1):
    # Extract the segment end point
    segment_end = segment_end[0, :3]
    segment_start = segment_start[0, :3]

    # Define the ray from segment start to end
    ray_origin = segment_start
    ray_direction = segment_end - segment_start
    ray_direction /= np.linalg.norm(ray_direction)

    # Perform ray-mesh intersection
    try:
        # Perform ray-mesh intersection
        result = mesh.ray.intersects_location(ray_origins=[ray_origin],
                                              ray_directions=[ray_direction],
                                              multiple_hits=False)
    except Exception as e:
        return False

    # Check if intersection is found at the segment_end point
    if len(result[0]) > 0:
        intersection_point = result[0][0]  # Get the first intersection point
        face_index = result[2][0]  # Get the index of the intersected face

        # Get the intersected face normal
        face_normal = mesh.face_normals[face_index]

        # Compute the dot product between the direction vector and face normal
        dot_product = np.dot(ray_direction, face_normal)

        # print(abs(dot_product))
        if abs(dot_product) > (1 - threshold) and mesh.contains([segment_end])[0]:
            # print('Direction is almost parallel to the surface!')

            return True

    return False


def is_segment_intersecting_mesh_1(segment_start, segment_end, mesh, threshold=0.1):
    # Extract the segment end point
    segment_end = segment_end[0, :3]
    segment_start = segment_start[0, :3]

    # Define the ray from segment start to end
    ray_origin = segment_start
    ray_direction = segment_end - segment_start
    ray_direction /= np.linalg.norm(ray_direction)

    # Perform ray-mesh intersection
    result = mesh.ray.intersects_location(ray_origins=[ray_origin],
                                          ray_directions=[ray_direction],
                                          multiple_hits=False)

    # Check if intersection is found at the segment_end point
    if len(result[0]) > 0:
        # print('Intersection found!')
        intersection_point = result[0][0]  # Get the first intersection point
        face_index = result[2][0]  # Get the index of the intersected face

        # Get the intersected face normal

        face_normal = mesh.face_normals[face_index]

        # Compute the dot product between the direction vector and face normal
        dot_product = np.dot(ray_direction, face_normal)
        # print(abs(dot_product))
        # Create a path representing the normal vector

        normal_path = path.Path3D([intersection_point, intersection_point + face_normal])

        # Add the normal vector as a line segment to the mesh visualization
        scene = mesh.scene()
        scene.add_geometry(normal_path)

        # Display the mesh with the normal vector
        scene.show()
        if abs(dot_product) < threshold and mesh.contains([segment_end])[0]:
            # print('Direction is almost perpendicular to the surface!')
            # Highlight the intersected face

            return True

    return False


# This function computes 2D to 3D projection
def to_3d(camera_pose, camera_intrinsics, point, depth, radius, res):
    # Camera intrinsics (width, height, fx, fy, cx, cy, distortion parameters)
    # Extract camera parameters

    # print(point)
    W = camera_intrinsics[0]
    H = camera_intrinsics[1]

    fx = camera_intrinsics[2]
    fy = camera_intrinsics[3]
    cx = W / 2.0
    cy = H / 2.0

    # Convert the 2D point to normalized image coordinates

    tx, ty = get_resolution(point, res, 456, 256)  # get_resolution function not defined in the given code

    # Convert the depth image to 3D coordinates
    # Y, X = np.where(depth > 0)
    Y, X = np.array([ty]), np.array([tx])

    Z = depth[Y, X]
    if abs(Z) == 0:
        Z = Z + 1
    if radius != None:
        radius = Z * (radius + X - cx) / fx

    x = Z * (X - cx) / fx
    y = Z * (Y - cy) / fy

    radius = radius - x

    # Create a matrix with the 3D coordinates
    uv_norm = np.ones((Z.shape[0], 4))
    uv_norm[:, 0] = x
    uv_norm[:, 1] = y
    uv_norm[:, 2] = Z

    # Convert the 3D coordinates from camera frame to world coordinate system using the camera pose
    pred_t = np.matmul(camera_pose, uv_norm.T).T
    pred_t = pred_t / pred_t[:, 3:4]  # Normalize the coordinates by dividing by the homogeneous coordinate
    return pred_t, radius


def to_2d(camera_pose, camera_intrinsics, pred_t, radius_3d):
    # Camera intrinsics (width, height, fx, fy, cx, cy, distortion parameters)
    # Extract camera parameters
    # print('pred_t: ', pred_t)
    W = camera_intrinsics[0]
    H = camera_intrinsics[1]
    fx = camera_intrinsics[2]
    fy = camera_intrinsics[3]
    cx = W / 2.0
    cy = H / 2.0

    # Convert the 3D coordinates from world coordinate system to camera frame
    pred_t_homogeneous = np.ones((pred_t.shape[0], 4))
    pred_t_homogeneous[:, :3] = pred_t[:, :3]
    pred_t_camera = np.matmul(np.linalg.inv(camera_pose), pred_t_homogeneous.T).T
    pred_t_camera = pred_t_camera / pred_t_camera[:,
                                    3:4]  # Normalize the coordinates by dividing by the homogeneous coordinate

    # Convert the 3D coordinates to 2D normalized image coordinates
    tx = pred_t_camera[:, 0] * fx / pred_t_camera[:, 2] + cx
    ty = pred_t_camera[:, 1] * fy / pred_t_camera[:, 2] + cy

    # Scale the 2D coordinates to the image dimensions
    # tx *= W
    # ty *= H
    tx = tx.astype(int)
    ty = ty.astype(int)
    # tx, ty = get_resolution_1((tx, ty), 854, 480)
    # Convert the 2D coordinates to integers
    #if radius_3d != None:
        # radius *= 2
        # radius = Z * (radius - cx) / fx
        #radius = (radius_3d * fx) / pred_t_camera[:, 2] + cx

    # Apply depth information
    tz = 1 / pred_t_camera[:, 2]  # Inverse of depth (1 / Z)

    return (tx, ty)


def project_to_2d(point_3d, camera_matrix):
    # Apply projection matrix
    point_2d_homogeneous = np.dot(camera_matrix, point_3d.T)

    # Divide by the last coordinate to obtain 2D coordinates
    point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
    # print('2D: ', point_2d)
    # Multiply by image dimensions
    x = point_2d[1] * 256  # Multiply by height (256)
    y = point_2d[0] * 456  # Multiply by width (456)

    return (x[0].tolist(), y[0].tolist())


def get_depth_shared(camera_pose, scene, camera_node, renderer):
    # Convert the camera pose from world to camera coordinates
    c2w_pose = camera_pose
    w2c = np.linalg.inv(c2w_pose)

    # Squash the first dimension if it is 1
    if w2c.ndim == 3 and w2c.shape[0] == 1:
        w2c = w2c[0]

    # Convert the camera pose to OpenGL coordinate system
    w2c_pose_opengl = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]) @ w2c
    c2w_pose_opengl = np.linalg.inv(w2c_pose_opengl)

    # Update the camera in the scene with the new OpenGL camera pose
    scene.set_pose(camera_node, pose=c2w_pose_opengl)

    # Render the scene and extract the depth map
    depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

    # Return the depth map
    return depth


# This function calculates the depth of a mesh given a camera pose
def get_depth(camera_pose, camera, image_size, mesh):
    # Create a Pyrender scene and add the mesh to it
    ''' That would be moved outside! '''
    scene = pyrender.Scene()

    pmesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(pmesh)

    # Convert the camera pose from world to camera coordinates
    c2w_pose = camera_pose
    w2c = np.linalg.inv(c2w_pose)[0]

    # Convert the camera pose to OpenGL coordinate system
    w2c_pose_opengl = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]) @ w2c
    c2w_pose_opengl = np.linalg.inv(w2c_pose_opengl)
    # Add the camera to the scene with the OpenGL camera pose
    scene.add(camera, pose=c2w_pose_opengl)  # update with new pose
    # Render the scene and extract the depth map
    r = pyrender.OffscreenRenderer(image_size[0], image_size[1])
    color, depth = r.render(scene)

    # Return the depth map
    return depth


def get_prediction_interval(y, y_hat, x, x_hat):
    n = y.size
    resid = y - y_hat
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))  # standard deviation of the error
    t = stats.t.ppf(0.975, n - 2)  # used for CI and PI bands
    pi = t * s_err * np.sqrt(1 + 1 / n + (x_hat - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    return pi


def get_resolution(point, res, new_res_0, new_res_1):
    # original resolution
    original_resolution_x = res[0]  # 854
    original_resolution_y = res[1]  # 480

    # target resolution
    target_resolution_x = new_res_0
    target_resolution_y = new_res_1

    # original pixel coordinates
    original_x = point[0]
    original_y = point[1]

    # calculate new pixel coordinates
    new_x = int(original_x * (target_resolution_x / original_resolution_x))
    new_y = int(original_y * (target_resolution_y / original_resolution_y))

    return new_x, new_y


def get_resolution_1(point, new_res_0, new_res_1):
    # original resolution
    original_resolution_x = 456
    original_resolution_y = 256

    # target resolution
    target_resolution_x = new_res_0
    target_resolution_y = new_res_1

    # original pixel coordinates
    original_x = point[0]
    original_y = point[1]

    # calculate new pixel coordinates
    new_x = int(original_x * (target_resolution_x / original_resolution_x))
    new_y = int(original_y * (target_resolution_y / original_resolution_y))

    return new_x, new_y


def convert_to_frame_number(number):
    frame_number = str(number).zfill(10)  # Pad with zeros to make it 10 digits
    frame_string = "frame_" + frame_number
    return frame_string


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def get_c2w(img_data: list) -> np.ndarray:
    """
    Args:
        img_data: list, [qvec, tvec] of w2c

    Returns:
        c2w: np.ndarray, 4x4 camera-to-world matrix
    """
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(img_data[:4])
    w2c[:3, -1] = img_data[4:7]
    c2w = np.linalg.inv(w2c)
    return c2w


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
        ]
    ])


def get_camera_pose_1(camera_poses, chosen_frame):
    return camera_poses[chosen_frame + '.jpg']


def find_bottom_point(vertices):
    bottom_point = vertices[0]
    for vertex in vertices:
        if vertex[1] >= bottom_point[1]:
            bottom_point = vertex
        elif vertex[1] == bottom_point[1] and vertex[0] < bottom_point[0]:
            bottom_point = vertex
    return bottom_point


def find_top_point(vertices):
    top_point = vertices[0]
    for vertex in vertices:
        if vertex[1] <= top_point[1]:
            top_point = vertex
        elif vertex[1] == top_point[1] and vertex[0] > top_point[0]:
            top_point = vertex
    return top_point


def get_bounding_box(vertices):
    x_coordinates = [vertex[0] for vertex in vertices]
    y_coordinates = [vertex[1] for vertex in vertices]
    min_x = min(x_coordinates)
    max_x = max(x_coordinates)
    min_y = min(y_coordinates)
    max_y = max(y_coordinates)
    return min_x, min_y, max_x, max_y


def get_radius_from_bbox(bbox):
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y
    radius = 0.5 * (math.sqrt((width ** 2) + (height ** 2)))
    radius /= 2
    return radius


def get_avg_h_w_from_bbox(bbox):
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y
    radius = (width + height) / 2
    return radius

def get_object_bbs_seg(data):
    bbs_dict = {}
    for entry in data:
        image_path = entry['image']['image_path']
        frame_name = os.path.splitext(os.path.basename(image_path))[0]
        p_list = []
        obj_list = []
        annotation_list = []
        annotations = entry['annotations']
        for annotation in annotations:
            annotation_seg = []
            if 'left' not in annotation['name'] and 'right' not in annotation['name']:
                p_list.append(get_bounding_box([item for sublist in annotation['segments'] for item in sublist]))
                obj_list.append(annotation['name'])
            if 'left hand' in annotation['name'] or 'right hand' in annotation['name']:
                for polygon in annotation['segments']:
                    for poly in polygon:
                        if poly == []:
                            poly = [[0.0, 0.0]]
                        annotation_seg.append(np.array(poly, dtype=np.int32))
            if len(annotation_seg) > 0:
                annotation_list.append(annotation_seg)
        bbs_dict[frame_name] = (annotation_list, p_list, obj_list)
    return bbs_dict


def get_object_bbs_new(data):
    bbs_dict = {}
    all_objs = []
    for entry in data:
        image_path = entry['image']['image_path']
        frame_name = os.path.splitext(os.path.basename(image_path))[0]
        p_list = []
        obj_list = []
        annotations = entry['annotations']
        for annotation in annotations:
            if annotation['name'] not in all_objs:
                all_objs.append(annotation['name'])
            if 'left hand' not in annotation['name'] and 'right hand' not in annotation['name']:
                p_list.append(get_bounding_box([item for sublist in annotation['segments'] for item in sublist]))
                obj_list.append(annotation['name'])
        if frame_name not in bbs_dict.keys():
            bbs_dict[frame_name] = (p_list, obj_list)
        else:
            # Get the current lists from the dictionary
            current_p_list, current_obj_list = bbs_dict[frame_name]

            # Extend the lists with new elements that are not already present
            for j, obj in enumerate(obj_list):
                if obj not in current_obj_list:
                    current_p_list.append(p_list[j])
                    current_obj_list.append(obj)

            # Update the dictionary with the extended lists
            bbs_dict[frame_name] = (current_p_list, current_obj_list)
    print(all_objs)
    return bbs_dict


def get_names(data):
    bbs_dict = {}
    all_objs = []
    for entry in data:
        image_path = entry['image']['image_path']
        frame_name = os.path.splitext(os.path.basename(image_path))[0]
        obj_list = []
        annotations = entry['annotations']
        for annotation in annotations:
            if annotation['name'] not in all_objs:
                all_objs.append(annotation['name'])
            if 'left' not in annotation['name'] and 'right' not in annotation['name']:
                obj_list.append(annotation['name'])
        if frame_name not in bbs_dict.keys():
            bbs_dict[frame_name] = obj_list
        else:
            # Get the current lists from the dictionary
            current_obj_list = bbs_dict[frame_name]

            # Extend the lists with new elements that are not already present
            for j, obj in enumerate(obj_list):
                if obj not in current_obj_list:
                    current_obj_list.append(obj)

            # Update the dictionary with the extended lists
            bbs_dict[frame_name] = current_obj_list
    return bbs_dict


def ms(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2 * np.pi:resolution * 2j, 0:np.pi:resolution * 1j]
    X = radius * np.cos(u) * np.sin(v) + x
    Y = radius * np.sin(u) * np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)


def masks_overlap(mask1, mask2):
    # Create black images with white polygons

    polygons = []
    polygons.append(mask1)
    ps1 = []
    for polygon in polygons:
        for poly in polygon:
            if poly == []:
                poly = [[0.0, 0.0]]
            ps1.append(np.array(poly, dtype=np.int32))

    polygons = []
    polygons.append(mask2)
    ps2 = []
    for polygon in polygons:
        for poly in polygon:
            if poly == []:
                poly = [[0.0, 0.0]]
            ps2.append(np.array(poly, dtype=np.int32))
    img1 = np.zeros((1080, 1920), dtype=np.uint8)
    img2 = np.zeros((480, 854), dtype=np.uint8)
    cv2.fillPoly(img1, ps1, (255, 255, 255))
    img1 = cv2.resize(img1, (854,
                             480),
                      interpolation=cv2.INTER_NEAREST)
    cv2.fillPoly(img2, ps2, (255, 255, 255))

    # Compute intersection and union
    intersection = cv2.bitwise_and(img1, img2)
    union = cv2.bitwise_or(img1, img2)

    # Compute overlap percentage
    if cv2.countNonZero(union) == 0:
        return 0
    overlap_percentage = cv2.countNonZero(intersection) / cv2.countNonZero(union)
    return overlap_percentage >= 0.2


def is_near(filename, lst, obj, contact_list, mask):
    # Find frame number in the dictionary which is closest to the frame number of the given filename, but not greater than 30
    # Show the image with the object

    frame_num = int(filename.split('_')[-1].split('.')[0])
    closest_frame_list = []
    for i in range(frame_num - 200, frame_num + 200):
        if i in lst:
            closest_frame_list = closest_frame_list + [i]

    # If a frame number is found, check if the corresponding element has the same list_obj_id as the given filename
    for c in closest_frame_list:

        closest_element = contact_list[c]
        if obj == closest_element[0]:

            if masks_overlap(closest_element[1], mask):
                return True

    return False


def draw_bounding_box(image, bbox_coordinates):
    # Load the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a copy of the image to draw on
    image = np.copy(image)

    # print(bbox_coordinates)
    # Extract the bounding box coordinates
    x1, y1, x2, y2 = bbox_coordinates

    # Draw the bounding box on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the image with the bounding box
    plt.imshow(image)
    plt.axis('off')
    plt.show()


transform = T.Compose([
    T.Resize((224, 224)),  # Resize the patch to a fixed size
    T.ToTensor(),  # Convert the patch to a tensor
    T.Normalize(  # Normalize the patch
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_dino_features_batch(images, bounding_boxes, model, device):
    # Define a function to process a single bounding box and apply the transformation
    def process_bbox(i):
        patch = images[i].crop(bounding_boxes[i])
        return transform(patch)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        patches = list(executor.map(process_bbox, range(len(bounding_boxes))))

    # Convert the patches list to a tensor
    patches = torch.stack(patches)

    # Pass the patches through the DINO model to extract features
    with torch.no_grad():
        features = model(patches.to(device))
    return features


def extract_vit_features_batch(images, bounding_boxes, model, transform):
    # Define a function to process a single bounding box and apply the transformation
    def process_bbox(i):
        patch = images[i].crop(bounding_boxes[i])
        width, height = patch.size
        if width == 0 or height == 0 or width == 1 or height == 1:
            patch = Image.new('RGB', (10, 10))

        return patch

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        patches = list(executor.map(process_bbox, range(len(bounding_boxes))))

    # Convert the patches list to a tensor
    try:
        patches = transform(images=patches, return_tensors="pt")
    except:
        import pdb;
        pdb.set_trace()
    # Pass the patches through the DINO model to extract features
    with torch.no_grad():
        features = model(**patches.to('cuda'))
        features = features['pooler_output']
    return features


def extract_clip_features(images, bounding_boxes, model, device):
    # Define a function to process a single bounding box and apply the transformation
    def process_bbox(i):
        patch = images[i].crop(bounding_boxes[i])
        return transform(patch)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        patches = list(executor.map(process_bbox, range(len(bounding_boxes))))

    # Convert the patches list to a tensor
    patches = torch.stack(patches)

    # Pass the patches through the CLIP model to extract features
    with torch.no_grad():
        features = model.encode_image(patches.cuda())

    return features


def get_bounding_box_center(box):
    min_x, min_y, max_x, max_y = box
    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + (width / 2)
    center_y = min_y + (height / 2)
    return center_x, center_y


def extract_3d_features(bounding_boxes, objects, camera_pose, depth, camera_intrinsics, data_path, frame, res):
    # Iterate over the bounding boxes
    loc_3d = []
    r_3d = []
    loc_3d_obj = {}
    r3d_obj = {}
    for i, bbox in enumerate(bounding_boxes):
        center = get_bounding_box_center(bbox)
        radius = get_radius_from_bbox(bbox)
        loca_embedding, r3d = to_3d(camera_pose, camera_intrinsics, center, depth, radius, res)
        loc_3d.append(torch.from_numpy(loca_embedding))
        r_3d.append(torch.from_numpy(r3d))
        loc_3d_obj[objects[i]] = torch.from_numpy(loca_embedding)
        r3d_obj[objects[i]] = r3d
    loc_3d = torch.stack(loc_3d)
    r_3d = torch.stack(r_3d)
    return loc_3d, loc_3d_obj, r_3d


def extract_3d_features_height_width(bounding_boxes, objects, camera_pose, depth, camera_intrinsics, data_path, frame):
    # Iterate over the bounding boxes
    loc_3d = []
    r_3d = []
    loc_3d_obj = {}
    r3d_obj = {}
    for i, bbox in enumerate(bounding_boxes):
        center = get_bounding_box_center(bbox)
        radius = get_avg_h_w_from_bbox(bbox)
        loca_embedding, r3d = to_3d(camera_pose, camera_intrinsics, center, depth, radius)
        loc_3d.append(torch.from_numpy(loca_embedding))
        r_3d.append(torch.from_numpy(r3d))
        loc_3d_obj[objects[i]] = torch.from_numpy(loca_embedding)
        r3d_obj[objects[i]] = r3d
    loc_3d = torch.stack(loc_3d)
    r_3d = torch.stack(r_3d)
    return r3d_obj


def extract_3d_features_no_center(bounding_boxes, objects, camera_pose, depth, camera_intrinsics, data_path, frame):
    # Iterate over the bounding boxes
    loc_3d = []
    r_3d = []
    loc_3d_obj = {}
    r3d_obj = {}
    for i, bbox in enumerate(bounding_boxes):
        center = bbox

        radius = 1
        loca_embedding, r3d = to_3d(camera_pose, camera_intrinsics, center, depth, radius)
        loc_3d.append(torch.from_numpy(loca_embedding))
        r_3d.append(torch.from_numpy(r3d))
        loc_3d_obj[objects[i]] = torch.from_numpy(loca_embedding)
        r3d_obj[objects[i]] = r3d
    loc_3d = torch.stack(loc_3d)
    r_3d = torch.stack(r_3d)
    return loc_3d, loc_3d_obj, r_3d


def visualize_mask(image, mask, bbox, color, text,
                   alpha=0.5, show_border=True, border_alpha=0.8,
                   border_thick=2, border_color=None):
    """Visualizes a single binary mask."""

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # draw border with bbox
    cv2.rectangle(image,
                  (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                  color, border_thick)

    # image = image.astype(np.float32)

    # get top corner of mask
    x, y = bbox[0], bbox[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    ((txt_w, txt_h), _) = cv2.getTextSize(text, font, font_scale, 1)

    # Place text background.
    back_tl = int(x), int(y)
    back_br = int(x) + int(txt_w), int(y) + int(1.3 * txt_h)
    txt_tl = int(x), int(y) + int(1 * txt_h)

    # draw text on top of mask
    image = rect_with_opacity(image, back_tl, back_br, (255, 255, 255), 0.6)

    cv2.putText(image, text, txt_tl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

    return image


def get_colors():
    try:
        RGB_tuples = np.vstack(
            [np.loadtxt("assets/colors.txt", skiprows=1), np.random.uniform(0, 255, size=(10000, 3)), [[0, 0, 0]]])
        b = np.where(RGB_tuples == 0)
        RGB_tuples[b] = 1
    except:
        colormap = np.array(
            list(ColorDict(norm=255, mode='rgb', palettes_path="", is_grayscale=False, palettes='all').values()))
        RGB_tuples = np.vstack([colormap[1:, :3], np.random.uniform(0, 255, size=(10000, 3)), [[0, 0, 0]]])

    return RGB_tuples


def rect_with_opacity(image, top_left, bottom_right, fill_color, fill_opacity):
    with_fill = image.copy()
    with_fill = cv2.rectangle(with_fill, top_left, bottom_right, fill_color, cv2.FILLED)
    return cv2.addWeighted(with_fill, fill_opacity, image, 1 - fill_opacity, 0, image)


def compute_tot_ids(results):
    total_tracked_ids = 0

    # Iterate over each frame in the results dictionary
    for frame_name in results:
        frame_data = results[frame_name]

        # Get the 'tracked_ids' list for the current frame
        tracked_ids = frame_data['tracked_ids']

        # Add the count of 'tracked_ids' to the total count
        total_tracked_ids += len(tracked_ids)
    return total_tracked_ids


def compute_similarity_matrix(tracked_ids, tracked_gt, tracked_bbox):
    num_detections = len(tracked_ids)
    num_gt_detections = len(tracked_gt)

    similarity_matrix = np.zeros((num_detections, num_gt_detections))

    for i in range(num_detections):
        for j in range(num_gt_detections):
            iou = calculate_iou(tracked_bbox[i], tracked_gt[j])
            similarity_matrix[i, j] = iou

    return similarity_matrix


def calculate_iou(bbox1, bbox2):
    # Calculate the intersection area
    bbox1 = convert_to_standard_format(bbox1)
    bbox2 = convert_to_standard_format(bbox2)
    intersection = calculate_intersection(bbox1, bbox2)

    # Calculate the union area
    union = calculate_union(bbox1, bbox2)

    # Calculate the IoU (Intersection over Union)
    iou = intersection / union

    return iou


def calculate_intersection(bbox1, bbox2):
    # Calculate the coordinates of the intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate the area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    return intersection


def calculate_union(bbox1, bbox2):
    # Calculate the area of bbox1
    area1 = calculate_area(bbox1)

    # Calculate the area of bbox2
    area2 = calculate_area(bbox2)

    # Calculate the area of union
    union = area1 + area2 - calculate_intersection(bbox1, bbox2)

    return union


def calculate_area(bbox):
    # Calculate the area of the bounding box
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height

    return area


def convert_to_standard_format(bbox):
    # Convert bbox format [x, y, w, h] to [x1, y1, x2, y2]
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]
    return [x1, y1, x2, y2]


def extract_frame_ground_truth(data):
    """Extract frame names and ground truth class IDs.

    Parameters
    ----------
    data : list of dict
        List of dictionaries containing the data.

    Returns
    -------
    frame_ground_truth : dict
        A dictionary mapping frame names to a list of ground truth class IDs.
    """
    frame_ground_truth = {}
    class_dict = {}
    for entry in data:
        image_path = entry['image']['image_path']
        frame_name = os.path.splitext(os.path.basename(image_path))[0]

        annotations = entry['annotations']
        class_ids = []
        for annotation in annotations:
            if 'left' not in annotation['name'] and 'right' not in annotation['name']:
                class_id = annotation['class_id']
                class_ids.append(class_id)
                if annotation['name'] not in class_dict.keys():
                    class_dict[annotation['name']] = class_id

        if len(class_ids) > 0:
            frame_ground_truth[frame_name] = list(set(class_ids))
            '''
            if frame_name in frame_ground_truth:
                frame_ground_truth[frame_name].extend(class_ids)
            else:
                frame_ground_truth[frame_name] = class_ids
            '''

    return frame_ground_truth, class_dict


def get_inverse_transform(s, R, t) -> torch.Tensor:
    """
    input matrix of {RsP + st} is [R, t, 1/s],
    and it's inverse is [R.T, -R.T @ s*t, s]
    """
    mat = np.eye(4)
    mat[:3, :3] = R.T
    mat[:3, 3] = - R.T @ (s * t)
    mat[-1, -1] = s
    return torch.from_numpy(mat).float()


def qtvec2mat(qvec: np.ndarray, tvec: np.ndarray) -> torch.Tensor:
    n = len(qvec)
    R = quaternion_to_matrix(torch.from_numpy(qvec))  # qvec2rotmat(qvec)
    mat = torch.eye(4).view(-1, 4, 4).tile([n, 1, 1])
    mat[:, :3, :3] = R
    mat[:, :3, 3] = torch.from_numpy(tvec)
    return mat.float()


def mat2qtvec(mat: torch.Tensor) -> np.ndarray:
    qvec = matrix_to_quaternion(mat[:, :3, :3]).numpy()
    tvec = mat[:, :3, 3].numpy()
    qtvecs = np.concatenate([qvec, tvec], 1)
    return qtvecs


def extract_values_from_json():
    scale = 1
    rot = np.array([
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0
    ]).reshape(3, 3)
    transl = np.array([
        0.0,
        0.0,
        0.0
    ])
    return scale, rot, transl


def get_camera_poses(path, kitchen, rescale):
    c2w_list = {}
    with open(path + '/poses.json', 'r') as f:
        model = json.load(f)
    #
    images = np.concatenate([np.expand_dims(np.array(values), axis=0) for key, values in model['images'].items()],
                            axis=0)
    # print(images.shape)
    w2c = qtvec2mat(images[:, :4], images[:, 4:])
    scale, rot, transl = extract_values_from_json()
    if rescale:
        if rescale_scores[kitchen] != None:
            scale = scale * rescale_scores[kitchen]
            transl = transl * rescale_scores[kitchen]
    inv_transf = get_inverse_transform(s=scale, R=rot, t=transl / scale)
    new_w2c = w2c @ inv_transf
    new_images = mat2qtvec(new_w2c)
    for e, i in enumerate(model['images'].keys()):
        c2w_list[i] = [get_c2w(new_images[e])]

    return c2w_list


def get_camera_poses_old_1(path):
    c2w_list = {}
    with open(path + '/poses.json', 'r') as f:
        model = json.load(f)

    for i in model['images'].keys():
        c2w_list[i] = [get_c2w(model['images'][i])]

    return c2w_list

def read_data_1(data_path, path_align, kitchen, rescale):
    path_interpolations = os.path.join(data_path, 'mask_annotations.json')
    file = open(path_interpolations)
    masks = json.load(file)
    mask_list = []
    for m in masks['video_annotations']:
        mask_list.append(m['image']['image_path'].split('/')[-1].split('.')[0].split('_')[2] + '_' +
                         m['image']['image_path'].split('/')[-1].split('.')[0].split('_')[3])
    annotations = {}
    class_dict = {}
    for m in masks['video_annotations']:
        img = m['image']['image_path'].split('/')[-1].split('.')[0].split('_')[2] + '_' + \
              m['image']['image_path'].split('/')[-1].split('.')[0].split('_')[3]
        annotations[img] = {}
        for e, a in enumerate(m['annotations']):
            annotations[img][a['class_id']] = a

            class_ids = []

            if 'left hand' not in a['name'] and 'right hand' not in a['name']:
                class_id = a['class_id']
                class_ids.append(class_id)
                if a['name'] not in class_dict.keys():
                    class_dict[a['name']] = class_id
    print(kitchen)
    camera_poses = get_camera_poses(data_path, kitchen, rescale)

    pose_avail = [k.split('.')[0] for k in camera_poses.keys()]
    common_pose_mask = set(pose_avail).intersection(mask_list)

    frames = list(common_pose_mask)
    # class_dict = {value: key for key, value in class_dict.items()}

    return masks, annotations, camera_poses, frames, class_dict

def read_data_all(data_path, kitchen, rescale):
    path_interpolations = os.path.join(data_path, 'mask_annotations.json')
    file = open(path_interpolations)
    masks = json.load(file)
    mask_list = []
    for m in masks['video_annotations']:
        mask_list.append(m['image']['image_path'].split('/')[-1].split('.')[0].split('_')[2] + '_' +
                         m['image']['image_path'].split('/')[-1].split('.')[0].split('_')[3])
    annotations = {}
    class_dict = {}
    for m in masks['video_annotations']:
        img = m['image']['image_path'].split('/')[-1].split('.')[0].split('_')[2] + '_' + \
              m['image']['image_path'].split('/')[-1].split('.')[0].split('_')[3]
        annotations[img] = {}
        for e, a in enumerate(m['annotations']):
            annotations[img][a['class_id']] = a

            class_ids = []

            if 'left hand' not in a['name'] and 'right hand' not in a['name']:
                class_id = a['class_id']
                class_ids.append(class_id)
                if a['name'] not in class_dict.keys():
                    class_dict[a['name']] = class_id

    camera_poses = get_camera_poses(data_path, kitchen, rescale)

    pose_avail = [k.split('.')[0] for k in camera_poses.keys()]
    common_pose_mask = set(pose_avail).intersection(mask_list)

    frames = list(common_pose_mask)

    return masks, annotations, camera_poses, frames, class_dict


import numpy as np

def get_bbox_from_binary(img):
    # Load the binary mask image
    binary_mask = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # Find contours in the binary mask
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the minimum and maximum coordinates
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf

    # Loop over the contours to find the bounding box of all contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Update the minimum and maximum coordinates
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x + w > max_x:
            max_x = x + w
        if y + h > max_y:
            max_y = y + h
        # Optional: Draw the bounding box on each contour for visualization

    # Create the bounding box that contains all segments
    combined_bounding_box = (min_x, min_y, max_x, max_y)
    contains_inf = any(math.isinf(value) for value in combined_bounding_box)
    if contains_inf:
        return None
    return combined_bounding_box
