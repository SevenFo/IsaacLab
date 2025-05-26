
def dynamic_set_attr(obj: object, kwargs: dict, path: list):
    """Dynamically set attributes on an object from a nested dictionary."""
    if kwargs is None:
        return

    for k, v in kwargs.items():
        if hasattr(obj, k):
            attr = getattr(obj, k)
            if isinstance(v, dict) and hasattr(attr, "__dict__"):
                next_path = path.copy()
                next_path.append(k)
                dynamic_set_attr(attr, v, next_path)
            else:
                try:
                    current_val = getattr(obj, k)
                    if isinstance(
                        current_val, (int, float, bool, str)
                    ) and not isinstance(v, type(current_val)):
                        # Type conversion if needed
                        v = type(current_val)(v)
                    setattr(obj, k, v)
                    print(
                        f"Set {'.'.join(path + [k])} from {getattr(obj, k)} to {v}"
                    )
                except Exception as e:
                    print(
                        f"Error setting attribute {'.'.join(path + [k])}: {e}"
                    )
        else:
            print(f"Warning: Attribute {k} not found in {'.'.join(path)}")
