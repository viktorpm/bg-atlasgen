# Define the version number for the atlas generation code
__version__ = "0"

# Import necessary libraries and modules
import json
import multiprocessing as mp
import time
import zipfile
from pathlib import Path

import imio
import numpy as np
import xmltodict
from bg_atlasapi import utils
from bg_atlasapi.structure_tree_util import get_structures_tree
from rich.progress import track

# Import custom functions and classes from other modules
from bg_atlasgen.mesh_utils import Region, create_region_mesh
from bg_atlasgen.wrapup import wrapup_atlas_from_data

# Flag to enable or disable parallel processing
PARALLEL = True

# Function to download atlas files from a given URL and extract them
def download_atlas_files(download_dir_path, atlas_file_url, ATLAS_NAME):
    atlas_files_dir = download_dir_path / ATLAS_NAME

    # If the atlas files directory already exists, return its path
    if atlas_files_dir.exists():
        return atlas_files_dir

    # Check if there is an internet connection
    utils.check_internet_connection()

    # Define the name for the downloaded atlas file and its destination path
    download_name = ATLAS_NAME + "_atlas.zip"
    destination_path = download_dir_path / download_name

    # Retrieve the atlas file over HTTP and save it to the destination path
    utils.retrieve_over_http(atlas_file_url, destination_path)

    # Extract the contents of the downloaded zip file to the atlas files directory
    with zipfile.ZipFile(destination_path, "r") as zip_ref:
        zip_ref.extractall(atlas_files_dir)

    return atlas_files_dir

# Function to parse structures from an XML file and build a hierarchical structure
def parse_structures_xml(root, path=None, structures=None):
    structures = structures or []
    path = path or []

    # Extract RGB color values from the XML and convert them to a tuple
    rgb_triplet = tuple(int(root["@color"][i : i + 2], 16) for i in (1, 3, 5))
    id = int(root["@id"])
    struct = {
        "name": root["@name"],
        "acronym": root["@abbreviation"],
        "id": int(root["@id"]),
        "structure_id_path": path + [id],
        "rgb_triplet": rgb_triplet,
    }
    structures.append(struct)

    # Recursively parse substructures if they exist
    if "label" in root:
        if isinstance(root["label"], list):
            for label in root["label"]:
                parse_structures_xml(
                    label, path=path + [id], structures=structures
                )
        else:
            parse_structures_xml(
                root["label"], path=path + [id], structures=structures
            )

    return structures

# Function to parse structures from an XML file
def parse_structures(structures_file: Path):
    root = xmltodict.parse(structures_file.read_text())["milf"]["structure"]
    root["@abbreviation"] = "root"
    root["@color"] = "#ffffff"
    root["@id"] = "10000"
    root["@name"] = "Root"

    structures = parse_structures_xml(root)
    return structures

# Function to create a hierarchical structure of regions
def create_structure_hierarchy(structures, df, root_id):
    for structure in structures:
        if structure["id"] != root_id:
            parent_id = structure["parent_structure_id"]
            while True:
                structure["structure_id_path"] = [parent_id] + structure[
                    "structure_id_path"
                ]
                if parent_id != root_id:
                    parent_id = int(
                        df[df["id"] == parent_id]["parent_structure_id"]
                    )
                else:
                    break
        else:
            structure["name"] = "root"
            structure["acronym"] = "root"

        del structure["parent_structure_id"]

    return structures

# Function to create 3D mesh representations of brain regions
def create_meshes(download_dir_path, tree, annotated_volume, labels, root_id):
    meshes_dir_path = download_dir_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False
        node.data = Region(is_label)

    # Mesh creation parameters
    closing_n_iters = 2
    decimate_fraction = 0.2
    smooth = False  # smooth meshes after creation
    start = time.time()

    # Check if parallel processing is enabled
    if PARALLEL:
        pool = mp.Pool(min(mp.cpu_count() - 2, 16))

        try:
            # Create meshes in parallel for all nodes in the tree
            pool.map(
                create_region_mesh,
                [
                    (
                        meshes_dir_path,
                        node,
                        tree,
                        labels,
                        annotated_volume,
                        root_id,
                        closing_n_iters,
                        decimate_fraction,
                        smooth,
                    )
                    for node in tree.nodes.values()
                ],
            )
        except mp.pool.MaybeEncodingError:
            pass
    else:
        # Create meshes sequentially and track progress using the rich library
        for node in track(
            tree.nodes.values(),
            total=tree.size(),
            description="Creating meshes",
        ):
            create_region_mesh(
                (
                    meshes_dir_path,
                    node,
                    tree,
                    labels,
                    annotated_volume,
                    root_id,
                    closing_n_iters,
                    decimate_fraction,
                    smooth,
                )
            )

    # Print the time taken for mesh extraction
    print(
        "Finished mesh extraction in: ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )
    return meshes_dir_path

# Function to create a dictionary of meshes for structures with valid meshes
def create_mesh_dict(structures, meshes_dir_path):
    meshes_dict = dict()
    structures_with_mesh = []
    for s in structures:
        # Check if a mesh was created for the structure
        mesh_path = meshes_dir_path / f'{s["id"]}.obj'
        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it")
            continue
        else:
            # Check that the mesh actually exists (i.e., not empty)
            if mesh_path.stat().st_size < 512:
                print(f"obj file for {s} is too small, ignoring it.")
                continue

        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        f"In the end, {len(structures_with_mesh)} structures with mesh are kept"
    )
    return meshes_dict, structures_with_mesh

# Function to create an atlas from downloaded data
def create_atlas(working_dir):
    ATLAS_NAME = "whs_sd_rat"
    SPECIES = "Rattus norvegicus"
    ATLAS_LINK = "https://www.nitrc.org/projects/whs-sd-atlas"
    CITATION = (
        "Papp et al 2014, https://doi.org/10.1016/j.neuroimage.2014.04.001"
    )
    ORIENTATION = "lpi"
    RESOLUTION = (39, 39, 39)
    ROOT_ID = 10000
    ATLAS_FILE_URL = "https://www.nitrc.org/frs/download.php/12263/MBAT_WHS_SD_rat_atlas_v4_pack.zip"
    ATLAS_PACKAGER = (
        "Ben Kantor, Tel Aviv University, Israel, benkantor@mail.tau.ac.il"
    )

    # Validate some properties of the atlas
    assert len(ORIENTATION) == 3, (
        "Orientation is not 3 characters, Got" + ORIENTATION
    )
    assert len(RESOLUTION) == 3, "Resolution is not correct, Got " + RESOLUTION
    assert (
        ATLAS_FILE_URL
    ), "No download link provided for atlas in ATLAS_FILE_URL"

    # Define the directory structure for the generated atlas
    working_dir = working_dir / ATLAS_NAME
    working_dir.mkdir(exist_ok=True, parents=True)

    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)

    # Download atlas files from the provided URL and extract them
    print("Downloading atlas from link: ", ATLAS_FILE_URL)
    atlas_files_dir = download_atlas_files(
        download_dir_path, ATLAS_FILE_URL, ATLAS_NAME
    )
    atlas_files_dir = atlas_files_dir / "MBAT_WHS_SD_rat_atlas_v4_pack/Data"

    # Parse structure metadata from an XML file
    structures = parse_structures(
        atlas_files_dir / "WHS_SD_rat_atlas_v4_labels.ilf"
    )

    # # Load annotation and reference stacks
    # annotation_stack = imio.load_any(
    #     atlas_files_dir / "WHS_SD_rat_atlas_v4.nii.gz", as_numpy=True
    # ).astype(np.int64)
    # reference_stack = imio.load_any(
    #     atlas_files_dir / "WHS_SD_rat_T2star_v1.01.nii.gz", as_numpy=True
    # )

    ###############################################################################
    # Load annotation and reference stacks at full resolution
    annotation_stack = imio.load_any(
        atlas_files_dir / "WHS_SD_rat_atlas_v4.nii.gz", as_numpy=True
    ).astype(np.int64)
    reference_stack = imio.load_any(
        atlas_files_dir / "WHS_SD_rat_T2star_v1.01.nii.gz", as_numpy=True
    )

    # Define the downsampling factor (e.g., reduce the data to 10%)
    downsample_factor = 0.1

    # Downsample the annotation stack
    annotation_stack = annotation_stack[
        :: int(1 / downsample_factor),
        :: int(1 / downsample_factor),
        :: int(1 / downsample_factor),
    ]

    # Downsample the reference stack (if needed)
    reference_stack = reference_stack[
        :: int(1 / downsample_factor),
        :: int(1 / downsample_factor),
        :: int(1 / downsample_factor),
    ]
    ############################################################################


    # Remove structures with missing annotations from the list of structures
    tree = get_structures_tree(structures)
    labels = set(np.unique(annotation_stack).astype(np.int32))
    existing_structures = []
    for structure in structures:
        stree = tree.subtree(structure["id"])
        ids = set(stree.nodes.keys())
        matched_labels = ids & labels
        if matched_labels:
            existing_structures.append(structure)
        else:
            node = tree.nodes[structure["id"]]
            print(
                f"{node.tag} not found in annotation volume, removing from list of structures..."
            )
    structures = existing_structures
    tree = get_structures_tree(structures)

    # Clean junk from the reference file based on annotations
    reference_stack *= annotation_stack > 0

    # Create a stack to represent hemispheres
    hemispheres_stack = np.full(reference_stack.shape, 2, dtype=np.uint8)
    hemispheres_stack[:244] = 1

    # Save the list of regions (structures) as a JSON file
    with open(download_dir_path / "structures.json", "w") as f:
        json.dump(structures, f)

    ###########################################################################
    # # Create 3D meshes for the brain regions
    # print(f"Saving atlas data at {download_dir_path}")
    # meshes_dir_path = create_meshes(
    #     download_dir_path, tree, annotation_stack, labels, ROOT_ID
    # )
    ############################################################################


    # Create 3D meshes for the downsampled data
    print(f"Saving atlas data at {download_dir_path}")
    meshes_dir_path = create_meshes(
        download_dir_path, tree, annotation_stack, labels, ROOT_ID
    )


    # Create a dictionary of meshes for structures with valid meshes
    meshes_dict, structures_with_mesh = create_mesh_dict(
        structures, meshes_dir_path
    )

    # Wrap up the atlas, compress it, and remove temporary files if required
    print("Finalising atlas")
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=RESOLUTION,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_stack,
        annotation_stack=annotation_stack,
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        atlas_packager=ATLAS_PACKAGER,
        hemispheres_stack=hemispheres_stack,
        cleanup_files=False,  # Set to True to remove temporary files
        compress=True,
        scale_meshes=True,
    )

    return output_filename

if __name__ == "__main__":
    # Define the root directory where the atlas will be generated
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    # Call the create_atlas function to generate the atlas
    create_atlas(bg_root_dir)
