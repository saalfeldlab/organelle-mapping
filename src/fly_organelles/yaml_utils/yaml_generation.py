#%%
import os
import glob
import yaml

import zarr

def get_right_resolutions(path, resolutions):
    z = zarr.open(path, mode='r')
    if not isinstance(z, zarr.hierarchy.Group):
        raise Exception(f"{path} is not a group")
    for i,res in enumerate(z.attrs.asdict()["multiscales"][0]["datasets"]):
        current_res = int(res["coordinateTransformations"][0]["scale"][0])
        if current_res == resolutions:
            return os.path.join(path,f"s{i}")
    raise Exception(f"{path} does not have resolution {resolutions}")

def detect_crops_from_gt_pattern(gt_pattern):
    if '[CROP]' not in gt_pattern:
        return []

    prefix, suffix = gt_pattern.split('[CROP]', maxsplit=1)

    glob_pattern = prefix + "*" + suffix
    matched_paths = glob.glob(glob_pattern)
    
    crops_found = []
    for mp in matched_paths:
        if suffix and mp.endswith(suffix):
            mp_no_suffix = mp[:-len(suffix)]
        else:
            mp_no_suffix = mp
        
        if mp_no_suffix.startswith(prefix):
            replaced_part = mp_no_suffix[len(prefix):]
        else:
            replaced_part = mp_no_suffix
        
        replaced_part = replaced_part.strip(os.sep)

        # Only add non-empty replaced parts
        if replaced_part and replaced_part not in crops_found:
            crops_found.append(replaced_part)

    return sorted(crops_found)

def create_yaml_with_crops(input_yaml_path, output_yaml_path):
    result = {}
    with open(input_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'datasets' not in data or not isinstance(data['datasets'], dict):
        raise ValueError("The input YAML must have a top-level 'datasets' dictionary.")

    for dataset_name, dataset_info in data['datasets'].items():
        result[dataset_name] = {}
        result[dataset_name]["contrast"] = [dataset_info.get('raw_min'), dataset_info.get('raw_max')]
        result[dataset_name]["raw"] = dataset_info.get('raw_path')
        result[dataset_name]["crops"]= {}
        
        gt_pattern = dataset_info.get('gt_pattern')
        if gt_pattern and '[CROP]' in gt_pattern:
            crops_found = detect_crops_from_gt_pattern(gt_pattern)
            print(f"Found crops for dataset {dataset_name}: {crops_found}")
        
            for crop in crops_found:
                crop_gt_path = gt_pattern.replace('[CROP]', crop)
                result[dataset_name]["crops"][crop] = crop_gt_path

    result = {"datasets": result}
    print(result)
    with open(output_yaml_path, 'w') as f:
        yaml.safe_dump(result, f, sort_keys=False)

# ----------------------------------------------------------------------------
# Example usage (run this part as a script or adapt to your environment):
if __name__ == "__main__":
    # Modify these paths to your actual input/output
    input_path = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/preparation/yamls/datasets.yaml"
    output_path = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/preparation/yamls/datasets_crops.yaml"
    res = 8

    create_yaml_with_crops(input_path, output_path,8,8)
    print(f"Wrote updated YAML with detected crops to: {output_path}")

# %%

# %%
# EXAMPLE INPUT YAML
# datasets:
#   jrc_c-elegans-op50-1:
#     raw_path: /nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/em/fibsem-int16
#     raw_min: 5000
#     raw_max: 6000
#     gt_pattern: /nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/labels/groundtruth/[CROP]/all
#   jrc_c-elegans-bw-1:
#     raw_path: /nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/em/fibsem-int16
#     raw_min: 5000
#     raw_max: 6000
#     gt_pattern: /nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/labels/groundtruth/[CROP]/all
#   jrc_c-elegans-comma-1:
#     raw_path: /nrs/cellmap/data/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1_downscaled.zarr/recon-1/em/fibsem-uint8
#     raw_min: 0
#     raw_max: 250
#     gt_pattern: /nrs/cellmap/zubovy/temp/jrc_c-elegans-comma-1_gt_8nm/[CROP].zarr
