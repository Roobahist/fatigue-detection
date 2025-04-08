import gc
import inspect
import sys
from typing import List
from memory_profiler import profile


def format_size(size_in_bytes):
    """Converts bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0


def print_all_memory(local_vars, global_vars):  
    """Prints the memory usage of provided local and global variables, sorted by name.
        usage : print_all_memory(locals(), globals())
        usage : vm(locals(), globals())
    """

    print("--- Local Variables ---")
    sorted_locals = sorted(local_vars.items())
    for name, value in sorted_locals:
        size = sys.getsizeof(value)
        formatted_size = format_size(size)
        print(f"{name}: {formatted_size}")

    print("\n--- Global Variables ---")
    global_vars_to_print = {}
    for name, value in global_vars.items():
        if name not in local_vars and not name.startswith("__"):
            global_vars_to_print[name] = value

    sorted_globals = sorted(global_vars_to_print.items())
    for name, value in sorted_globals:
        size = sys.getsizeof(value)
        formatted_size = format_size(size)
        print(f"{name}: {formatted_size}")

# @profile
def cleanup(var_names: List[str] = [], local_vars=None, global_vars=None):
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
            print(f"deleting local {var_name}")
            del local_vars[var_name]

        if global_vars and var_name in global_vars:
            print(f"deleting global {var_name}")
            del global_vars[var_name]

    gc.collect()
