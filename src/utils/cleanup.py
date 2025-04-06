import gc
from typing import List

def cleanup(var_names:List[str] = [], local_vars=None, global_vars=None):
    """
    Cleans up variables safely:
    - Calls .release() or .close() if available
    - Deletes variable from provided locals/globals
    - Triggers garbage collection
    """
    for var_name in var_names:
        var = None
        if local_vars and var_name in local_vars:
            var = local_vars[var_name]
        elif global_vars and var_name in global_vars:
            var = global_vars[var_name]

        if var is not None:
            try:
                # Release external resources (like cv2.VideoCapture, VideoWriter, files)
                if hasattr(var, "release"):
                    var.release()
                elif hasattr(var, "close"):
                    var.close()
            except Exception as e:
                pass  # Not all objects cleanly release/close

        # Delete from local or global scope
        if local_vars and var_name in local_vars:
            del local_vars[var_name]
        if global_vars and var_name in global_vars:
            del global_vars[var_name]

    gc.collect()
