
import zarr
import os
import numpy

def separate_store_path(store, path):
    """
    sometimes you can pass a total os path to node, leading to
    an empty('') node.path attribute.
    the correct way is to separate path to container(.n5, .zarr)
    from path to array within a container.

    Args:
        store (string): path to store
        path (string): path array/group (.n5 or .zarr)

    Returns:
        (string, string): returns regularized store and group/array path
    """
    new_store, path_prefix = os.path.split(store)
    if ".zarr" in path_prefix:
        return store, path
    return separate_store_path(new_store, os.path.join(path_prefix, path))

def access_parent(node):
    """
    Get the parent (zarr.Group) of an input zarr array(ds).


    Args:
        node (zarr.core.Array or zarr.hierarchy.Group): _description_

    Raises:
        RuntimeError: returned if the node array is in the parent group,
        or the group itself is the root group

    Returns:
        zarr.hierarchy.Group : parent group that contains input group/array
    """

    store_path, node_path = separate_store_path(node.store.path, node.path)
    if node_path == "":
        raise RuntimeError(
            f"{node.name} is in the root group of the {node.store.path} store."
        )
    else:
        return zarr.open(store=store_path, path=os.path.split(node_path.rstrip('/'))[0], mode="a")
    
    
def insert_omero_metadata(src : str,
                          window_max : int = None,
                          window_min : int = None,
                          window_start : int = None,
                          window_end : int = None,
                          id : int = None,
                          name : str = None):
    """
    Insert or update missing omero transitional metadata into .zattrs metadata of parent group for the input zarr array.
    

    Args:
        src (str): Path to Zarr array.
        window_max (int, optional): Max view window value. Defaults to None.
        window_min (int, optional): Min view window value. Defaults to None.
        window_start (int, optional): Contrast min value. Defaults to None.
        window_end (int, optional): Contrast max value. Defaults to None.
        id (int, optional): Defaults to None.
        name (str, optional): Name of the dataset. Defaults to None.
    """
    
    store_path, zarr_path = separate_store_path(src, '')
    
    z_store = zarr.NestedDirectoryStore(store_path)
    z_arr = zarr.open(store=z_store, path=zarr_path, mode='a')
    
    parent_group = access_parent(z_arr)

    if window_max == None:   
        window_max = numpy.iinfo(z_arr.dtype).max
    if window_min == None:
        window_min = numpy.iinfo(z_arr.dtype).min
    
    omero = dict()
    omero['id'] = 1 if id==None else id
    omero['name'] = os.path.basename(z_store.path.rstrip('/')).split('.')[0] if name==None else name
    omero['version'] = "0.4"            
    omero['channels'] = [                        
        {
            "active": True,
            "coefficient": 1,
            "color": "FFFFFF",
            "inverted": False,
            "label": parent_group.path.split('/')[-1],
            "window": {
                "end": window_max if window_end==None else window_end,
                "max": window_max,
                "min": window_min,
                "start":window_min if window_start==None else window_start
            }
        }
    ]
    omero["rdefs"] = {
        "defaultT": 0, 
        "defaultZ": int(z_arr.shape[0]/2),                 
        "model": "greyscale"  
    }     
    parent_group.attrs['omero'] = omero
    
