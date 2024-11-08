import json
import trimesh
import numpy as np
from trimesh.smoothing import filter_humphrey, filter_laplacian


def extract_faces_for_tooth(json_file_path, tooth_label):
    """
    Open a JSON file and extract face indices for a specified tooth label.

    Args:
    - json_file_path (str): Path to the JSON annotation file.
    - tooth_label (int): The label of the tooth to extract.

    Returns:
    - list of int: Indices of faces representing the specified tooth.
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file) 
        faces_for_tooth = [index for index, label in enumerate(data["labels"]) if label == tooth_label]
        return faces_for_tooth
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error: {e}")
        return []

def make_tooth_taller(mesh, json_file_path, tooth_label, height_factor):
    """
    Modify the shape of a specific tooth to make it taller.

    Args:
    - mesh (trimesh.Trimesh): The mesh object.
    - json_file_path (str): Path to the JSON annotation file.
    - tooth_label (int): The label of the tooth to modify.
    - height_factor (float): Scaling factor for the Z-axis to increase tooth height.

    Returns:
    - trimesh.Trimesh: The modified mesh.
    """
    # Extract faces for the specified tooth
    faces_for_tooth = extract_faces_for_tooth(json_file_path, tooth_label)
    if not faces_for_tooth:
        print("No faces found for the specified tooth.")
        return None
    tooth_faces = mesh.faces[faces_for_tooth]
    tooth_vertices = np.unique(tooth_faces.flatten())
    # Scale only the Z-axis of the vertices to make the tooth taller
    mesh.vertices[tooth_vertices][:, 2] *= height_factor  # Z-axis scaling
    return mesh

def resize_tooth(mesh, tooth_vertices_indices, scale_factor):
    """
    Resize a specific tooth in the original mesh by scaling the vertices corresponding to the tooth.

    Parameters:
    - mesh (trimesh.Trimesh): The mesh object.
    - tooth_vertices_indices (list): Indices of the vertices representing the tooth to be resized.
    - scale_factor (float): The factor by which to resize the tooth.

    Returns:
    - trimesh.Trimesh: The modified mesh.
    """
    tooth_vertices = mesh.vertices[tooth_vertices_indices]
    tooth_center = tooth_vertices.mean(axis=0)

    # Translate the tooth vertices to the origin (for scaling)
    translated_vertices = tooth_vertices - tooth_center
    # Apply the scaling factor to the translated vertices
    scaled_vertices = translated_vertices * scale_factor
    # Translate the vertices back to their original position
    final_scaled_vertices = scaled_vertices + tooth_center
    mesh.vertices[tooth_vertices_indices] = final_scaled_vertices
    return mesh

def translate_region(mesh, vertex_indices, translation_vector):
    """
    Translate a region of the mesh.

    Args:
    - mesh (trimesh.Trimesh): The mesh object.
    - vertex_indices (numpy.ndarray): Indices of vertices to translate.
    - translation_vector (list): Translation vector [tx, ty, tz].
    """
    mesh.vertices[vertex_indices] += translation_vector

def rotate_region(mesh, vertex_indices, angle, axis):
    """
    Rotate a region of the mesh around a specified axis.

    Args:
    - mesh (trimesh.Trimesh): The mesh object.
    - vertex_indices (numpy.ndarray): Indices of vertices to rotate.
    - angle (float): Rotation angle in degrees.
    - axis (list): Axis of rotation [x, y, z].
    """
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), axis)
    mesh.vertices[vertex_indices] = trimesh.transform_points(mesh.vertices[vertex_indices], rotation_matrix)


def smooth_sharpen_region(mesh, vertex_indices, smooth=True, iterations=10):
    """
    Apply smoothing or sharpening filter to a region of the mesh.

    Args:
    - mesh (trimesh.Trimesh): The mesh object.
    - vertex_indices (numpy.ndarray): Indices of vertices to smooth or sharpen.
    - smooth (bool): True to smooth, False to sharpen.
    - iterations (int): Number of iterations for the smoothing or sharpening.
    """
    region_faces = mesh.faces[vertex_indices]
    sub_mesh = trimesh.Trimesh(vertices=mesh.vertices[vertex_indices], faces=region_faces)   
    if smooth:
        filter_humphrey(sub_mesh, iterations=iterations)
    else:
        filter_laplacian(sub_mesh, iterations=iterations)  
    mesh.vertices[vertex_indices] = sub_mesh.vertices

def replace_tooth(mesh, replacement_tooth, target_tooth_vertices):
    """
    Replace a specific tooth in the original mesh with the replacement tooth model.

    Parameters:
    - mesh (trimesh.Trimesh): The original mesh object.
    - replacement_tooth (trimesh.Trimesh): The replacement tooth mesh object.
    - target_tooth_vertices (list): Indices of the vertices representing the tooth to be replaced.

    Returns:
    - trimesh.Trimesh: The mesh with the replaced tooth.
    """
    # Remove the target tooth vertices from the original mesh
    mask = np.ones(len(mesh.vertices), dtype=bool)
    mask[target_tooth_vertices] = False
    mesh.update_vertices(mask)
    # Compute the center of the target tooth region in the original mesh
    target_center = mesh.vertices[target_tooth_vertices].mean(axis=0)
    # Compute the center of the replacement tooth
    replacement_center = replacement_tooth.vertices.mean(axis=0)
    # Translate the replacement tooth to the target location
    translation_vector = target_center - replacement_center
    replacement_tooth.apply_translation(translation_vector)
    # Merge the replacement tooth into the original mesh
    combined_mesh = mesh + replacement_tooth
    return combined_mesh



# Load the original mesh
mesh = trimesh.load('path/mesh.obj') # put ur path
tooth_label = 22  # Example label for the tooth
json_file_path = 'path/patient_id.json'
# Make ur desired modification byr calling the appropriate functions
# Save the modified mesh
mesh.export('path/modified_mesh.obj')
