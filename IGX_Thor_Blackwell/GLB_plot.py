from pathlib import Path
import re
import traceback
import nibabel as nib
import numpy as np
import pyvista as pv
import trimesh
from skimage.measure import marching_cubes

# ============================================================
# 1. CHANGE THESE TWO PATHS
# ============================================================
INPUT_FOLDER = Path(
   r"C:\Users\2375353\OneDrive - Cognizant\Desktop\JnJ Heart 3D\ImageCAS-STACOM2025-02-10-2025\sample segmentations"
)
OUTPUT_ROOT = Path(
  r"C:\Users\2375353\OneDrive - Cognizant\Desktop\JnJ Heart 3D\ImageCAS-STACOM2025-02-10-2025\GLB Output"
)

# ============================================================
# 2. OUTPUT FOLDERS
# ============================================================
WHOLE_OUTPUT_FOLDER = (
   OUTPUT_ROOT / "01_Whole_Segmentation_GLB"
)
LA_PV_OUTPUT_FOLDER = (
   OUTPUT_ROOT / "02_LA_PV_GLB"
)
LA_PV_LAA_OUTPUT_FOLDER = (
   OUTPUT_ROOT / "03_LA_PV_LAA_GLB"
)
WHOLE_OUTPUT_FOLDER.mkdir(
   parents=True,
   exist_ok=True
)
LA_PV_OUTPUT_FOLDER.mkdir(
   parents=True,
   exist_ok=True
)
LA_PV_LAA_OUTPUT_FOLDER.mkdir(
   parents=True,
   exist_ok=True
)

# ============================================================
# 3. LABEL DEFINITIONS
# ============================================================
LABEL_NAMES = {
   1: "Myocardium",
   2: "Left_Atrium",
   3: "Left_Ventricle",
   4: "Right_Atrium",
   5: "Right_Ventricle",
   6: "Aorta",
   7: "Pulmonary_Artery",
   8: "Left_Atrial_Appendage",
   9: "Coronary_Arteries",
   10: "Pulmonary_Veins"
}

# RGBA colours: Red, Green, Blue, Alpha
LABEL_COLOURS = {
   1: [180, 80, 80, 255],       # Myocardium
   2: [220, 60, 60, 255],       # LA
   3: [190, 40, 80, 255],       # LV
   4: [70, 110, 220, 255],      # RA
   5: [70, 150, 230, 255],      # RV
   6: [230, 110, 80, 255],      # Aorta
   7: [130, 90, 210, 255],      # PA
   8: [240, 190, 50, 255],      # LAA
   9: [230, 160, 40, 255],      # Coronary
   10: [30, 180, 190, 255]      # PV
}

# ============================================================
# 4. PROCESSING SETTINGS
# ============================================================
# Skip a GLB if it already exists.
# Useful when restarting after interruption.
SKIP_EXISTING_FILES = True

# Display the first successfully processed dataset in Python.
PREVIEW_FIRST_DATASET = True

# Do not display every dataset because 998 interactive windows
# would need to be closed manually.
PREVIEW_EVERY_DATASET = False

# Preview which output:
# "whole"
# "la_pv"
# "la_pv_laa"
PREVIEW_TYPE = "la_pv_laa"

# Minimum number of voxels needed to create a surface.
# This avoids isolated one-voxel noise.
MINIMUM_VOXELS = 10

# Optional mesh reduction.
#
# 0.0 = no reduction and highest geometric detail.
# 0.3 = approximately 30% mesh reduction.
# 0.5 = approximately 50% mesh reduction.
#
# Keep at 0.0 initially to preserve the segmentation geometry.
MESH_REDUCTION = 0.0

# ============================================================
# 5. NATURAL NUMERIC SORTING
# ============================================================
def natural_sort_key(path_object):
   text = path_object.name
   return [
       int(part) if part.isdigit() else part.lower()
       for part in re.split(r"(\d+)", text)
   ]

# ============================================================
# 6. REMOVE .NII OR .NII.GZ EXTENSION
# ============================================================
def get_model_name(file_path):
   file_name = file_path.name
   if file_name.lower().endswith(".nii.gz"):
       return file_name[:-7]
   if file_name.lower().endswith(".nii"):
       return file_name[:-4]
   return file_path.stem

# ============================================================
# 7. CREATE ONE SURFACE FOR ONE LABEL
# ============================================================
def create_label_surface(
   segmentation_array,
   affine_matrix,
   label_value
):
   """
   Create a triangular 3D surface for one segmentation label.
   The NIfTI affine is applied to the vertices so that the
   GLB preserves the physical location, orientation and scale
   represented by the NIfTI file.
   """
   binary_mask = (
       segmentation_array == label_value
   )
   number_of_voxels = int(
       np.count_nonzero(binary_mask)
   )
   if number_of_voxels < MINIMUM_VOXELS:
       return None
   # Add a one-voxel background border.
   # This allows marching cubes to close structures that touch
   # the original image boundary.
   padded_mask = np.pad(
       binary_mask.astype(np.uint8),
       pad_width=1,
       mode="constant",
       constant_values=0
   )
   try:
       vertices, faces, normals, values = marching_cubes(
           padded_mask,
           level=0.5,
           allow_degenerate=False
       )
   except ValueError:
       return None
   # Remove the one-voxel padding offset.
   vertices = vertices - 1.0
   # Convert voxel coordinates to NIfTI physical coordinates.
   #
   # marching_cubes returns coordinates corresponding to the
   # three array axes. Applying the complete affine preserves:
   #
   # - voxel spacing
   # - orientation
   # - translation
   # - axis direction
   vertices_world = nib.affines.apply_affine(
       affine_matrix,
       vertices
   )
   mesh = trimesh.Trimesh(
       vertices=np.asarray(
           vertices_world,
           dtype=np.float32
       ),
       faces=np.asarray(
           faces,
           dtype=np.int64
       ),
       process=False
   )
   # Remove invalid and unused elements.
   mesh.remove_unreferenced_vertices()
   # Repair triangle orientation where possible.
   trimesh.repair.fix_normals(mesh)
   # Optional mesh simplification.
   if (
       MESH_REDUCTION > 0.0
       and len(mesh.faces) > 100
   ):
       target_face_count = int(
           len(mesh.faces) *
           (1.0 - MESH_REDUCTION)
       )
       target_face_count = max(
           target_face_count,
           100
       )
       try:
           mesh = mesh.simplify_quadric_decimation(
               face_count=target_face_count
           )
       except Exception:
           # Continue using the original mesh when the optional
           # simplification dependency is unavailable.
           pass
   colour = LABEL_COLOURS.get(
       label_value,
       [180, 180, 180, 255]
   )
   mesh.visual.face_colors = np.tile(
       np.asarray(
           colour,
           dtype=np.uint8
       ),
       (len(mesh.faces), 1)
   )
   return mesh

# ============================================================
# 8. CREATE AND SAVE ONE GLB SCENE
# ============================================================
def create_glb(
   segmentation_array,
   affine_matrix,
   labels_to_include,
   output_path
):
   """
   Create one GLB file containing separate objects for each
   requested anatomical label.
   """
   scene = trimesh.Scene()
   surfaces_added = 0
   for label_value in labels_to_include:
       mesh = create_label_surface(
           segmentation_array=segmentation_array,
           affine_matrix=affine_matrix,
           label_value=label_value
       )
       if mesh is None:
           continue
       structure_name = LABEL_NAMES.get(
           label_value,
           f"Label_{label_value}"
       )
       scene.add_geometry(
           geometry=mesh,
           node_name=structure_name,
           geom_name=structure_name
       )
       surfaces_added += 1
   if surfaces_added == 0:
       return None
   # Export the scene as one binary GLB file.
   glb_binary = trimesh.exchange.gltf.export_glb(
       scene
   )
   with open(output_path, "wb") as output_file:
       output_file.write(glb_binary)
   return scene

# ============================================================
# 9. PYVISTA INTERACTIVE VIEWER
# ============================================================
def display_scene_in_pyvista(
   scene,
   window_title
):
   """
   Display a Trimesh scene interactively in the Python IDE.
   Mouse controls:
   - Left drag: rotate
   - Mouse wheel: zoom
   - Right drag: pan
   """
   if scene is None:
       return
   plotter = pv.Plotter(
       window_size=(1200, 900),
       title=window_title
   )
   plotter.set_background("white")
   for geometry_name, geometry in scene.geometry.items():
       vertices = np.asarray(
           geometry.vertices
       )
       triangle_faces = np.asarray(
           geometry.faces
       )
       # PyVista face format:
       # [3, point1, point2, point3, 3, ...]
       pyvista_faces = np.column_stack(
           [
               np.full(
                   len(triangle_faces),
                   3,
                   dtype=np.int64
               ),
               triangle_faces
           ]
       ).ravel()
       pyvista_mesh = pv.PolyData(
           vertices,
           pyvista_faces
       )
       if hasattr(
           geometry.visual,
           "face_colors"
       ) and len(geometry.visual.face_colors) > 0:
           rgba = np.asarray(
               geometry.visual.face_colors[0],
               dtype=float
           )
           mesh_colour = (
               rgba[:3] / 255.0
           )
       else:
           mesh_colour = (
               0.7,
               0.7,
               0.7
           )
       plotter.add_mesh(
           pyvista_mesh,
           color=mesh_colour,
           smooth_shading=True,
           show_edges=False,
           name=str(geometry_name)
       )
   plotter.add_text(
       window_title,
       position="upper_left",
       font_size=10
   )
   # No grid lines or axes.
   plotter.hide_axes()
   plotter.camera_position = "iso"
   plotter.reset_camera()
   plotter.show()

# ============================================================
# 10. FIND ALL NIFTI FILES
# ============================================================
nifti_files = list(
   INPUT_FOLDER.rglob("*.nii.gz")
)
nifti_files.extend(
   INPUT_FOLDER.rglob("*.nii")
)
# Avoid duplicate entries if required.
nifti_files = sorted(
   set(nifti_files),
   key=natural_sort_key
)

if len(nifti_files) == 0:
   raise FileNotFoundError(
       f"No .nii or .nii.gz files were found in:\n"
       f"{INPUT_FOLDER}"
   )

print("=" * 70)
print("NIfTI TO GLB CONVERSION")
print("=" * 70)
print(f"Input folder       : {INPUT_FOLDER}")
print(f"Files found        : {len(nifti_files)}")
print(f"Output root        : {OUTPUT_ROOT}")
print("=" * 70)

# ============================================================
# 11. PROCESS ALL DATASETS
# ============================================================
successful_datasets = 0
failed_datasets = 0
preview_already_displayed = False

for file_number, nifti_path in enumerate(
   nifti_files,
   start=1
):
   model_name = get_model_name(
       nifti_path
   )
   whole_glb_path = (
       WHOLE_OUTPUT_FOLDER /
       f"{model_name}_whole.glb"
   )
   la_pv_glb_path = (
       LA_PV_OUTPUT_FOLDER /
       f"{model_name}_LA_PV.glb"
   )
   la_pv_laa_glb_path = (
       LA_PV_LAA_OUTPUT_FOLDER /
       f"{model_name}_LA_PV_LAA.glb"
   )
   print()
   print("-" * 70)
   print(
       f"[{file_number}/{len(nifti_files)}] "
       f"Processing: {nifti_path.name}"
   )
   try:
       # ----------------------------------------------------
       # Load segmentation
       # ----------------------------------------------------
       nifti_image = nib.load(
           str(nifti_path)
       )
       # np.asanyarray avoids unnecessarily loading the labels
       # as floating-point values through get_fdata().
       segmentation_array = np.asanyarray(
           nifti_image.dataobj
       )
       segmentation_array = np.rint(
           segmentation_array
       ).astype(
           np.int16,
           copy=False
       )
       affine_matrix = np.asarray(
           nifti_image.affine,
           dtype=np.float64
       )
       present_labels = sorted(
           int(value)
           for value in np.unique(segmentation_array)
           if int(value) != 0
       )
       print(
           f"  Image shape      : "
           f"{segmentation_array.shape}"
       )
       print(
           f"  Labels present   : "
           f"{present_labels}"
       )
       # ----------------------------------------------------
       # A. WHOLE SEGMENTATION
       # ----------------------------------------------------
       whole_scene = None
       if (
           SKIP_EXISTING_FILES
           and whole_glb_path.exists()
       ):
           print(
               "  Whole GLB        : already exists"
           )
       else:
           whole_labels = [
               label
               for label in present_labels
               if label > 0
           ]
           whole_scene = create_glb(
               segmentation_array=segmentation_array,
               affine_matrix=affine_matrix,
               labels_to_include=whole_labels,
               output_path=whole_glb_path
           )
           if whole_scene is None:
               print(
                   "  Whole GLB        : no surfaces"
               )
           else:
               print(
                   f"  Whole GLB        : "
                   f"{whole_glb_path.name}"
               )
       # ----------------------------------------------------
       # B. LA AND PV ONLY
       #
       # LA = 2
       # PV = 10
       # ----------------------------------------------------
       la_pv_scene = None
       if (
           SKIP_EXISTING_FILES
           and la_pv_glb_path.exists()
       ):
           print(
               "  LA + PV GLB      : already exists"
           )
       else:
           la_pv_scene = create_glb(
               segmentation_array=segmentation_array,
               affine_matrix=affine_matrix,
               labels_to_include=[2, 10],
               output_path=la_pv_glb_path
           )
           if la_pv_scene is None:
               print(
                   "  LA + PV GLB      : labels missing"
               )
           else:
               print(
                   f"  LA + PV GLB      : "
                   f"{la_pv_glb_path.name}"
               )
       # ----------------------------------------------------
       # C. LA, PV AND LAA
       #
       # LA  = 2
       # PV  = 10
       # LAA = 8
       # ----------------------------------------------------
       la_pv_laa_scene = None
       if (
           SKIP_EXISTING_FILES
           and la_pv_laa_glb_path.exists()
       ):
           print(
               "  LA+PV+LAA GLB    : already exists"
           )
       else:
           la_pv_laa_scene = create_glb(
               segmentation_array=segmentation_array,
               affine_matrix=affine_matrix,
               labels_to_include=[2, 10, 8],
               output_path=la_pv_laa_glb_path
           )
           if la_pv_laa_scene is None:
               print(
                   "  LA+PV+LAA GLB    : labels missing"
               )
           else:
               print(
                   f"  LA+PV+LAA GLB    : "
                   f"{la_pv_laa_glb_path.name}"
               )
       successful_datasets += 1
       # ----------------------------------------------------
       # Interactive Python preview
       # ----------------------------------------------------
       should_preview = (
           PREVIEW_EVERY_DATASET
           or (
               PREVIEW_FIRST_DATASET
               and not preview_already_displayed
           )
       )
       if should_preview:
           scene_to_preview = None
           preview_name = ""
           if PREVIEW_TYPE == "whole":
               scene_to_preview = whole_scene
               preview_name = "Whole segmentation"
               if (
                   scene_to_preview is None
                   and whole_glb_path.exists()
               ):
                   scene_to_preview = trimesh.load(
                       whole_glb_path,
                       force="scene"
                   )
           elif PREVIEW_TYPE == "la_pv":
               scene_to_preview = la_pv_scene
               preview_name = "LA and PV"
               if (
                   scene_to_preview is None
                   and la_pv_glb_path.exists()
               ):
                   scene_to_preview = trimesh.load(
                       la_pv_glb_path,
                       force="scene"
                   )
           else:
               scene_to_preview = la_pv_laa_scene
               preview_name = "LA, PV and LAA"
               if (
                   scene_to_preview is None
                   and la_pv_laa_glb_path.exists()
               ):
                   scene_to_preview = trimesh.load(
                       la_pv_laa_glb_path,
                       force="scene"
                   )
           if scene_to_preview is not None:
               display_scene_in_pyvista(
                   scene=scene_to_preview,
                   window_title=(
                       f"{model_name} – {preview_name}"
                   )
               )
               preview_already_displayed = True
       # Release the large image array before the next file.
       del segmentation_array
       del nifti_image
   except Exception as error:
       failed_datasets += 1
       print(
           f"  FAILED           : {error}"
       )
       traceback.print_exc()

# ============================================================
# 12. FINAL RESULT
# ============================================================
whole_count = len(
   list(
       WHOLE_OUTPUT_FOLDER.glob("*.glb")
   )
)
la_pv_count = len(
   list(
       LA_PV_OUTPUT_FOLDER.glob("*.glb")
   )
)
la_pv_laa_count = len(
   list(
       LA_PV_LAA_OUTPUT_FOLDER.glob("*.glb")
   )
)

print()
print("=" * 70)
print("CONVERSION FINISHED")
print("=" * 70)
print(
   f"Datasets attempted       : {len(nifti_files)}"
)
print(
   f"Datasets processed       : {successful_datasets}"
)
print(
   f"Datasets failed          : {failed_datasets}"
)
print(
   f"Whole GLB files          : {whole_count}"
)
print(
   f"LA + PV GLB files        : {la_pv_count}"
)
print(
   f"LA + PV + LAA GLB files  : {la_pv_laa_count}"
)
print()
print(
   f"Whole segmentation:\n"
   f"{WHOLE_OUTPUT_FOLDER}"
)
print()
print(
   f"LA and PV:\n"
   f"{LA_PV_OUTPUT_FOLDER}"
)
print()
print(
   f"LA, PV and LAA:\n"
   f"{LA_PV_LAA_OUTPUT_FOLDER}"
)
print("=" * 70)