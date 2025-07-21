"""
Skills package for the robot brain system.
"""

# Import skills to register them with the global registry
try:
    from . import basic_skills
    from . import manipulation_skills
    from . import observation_skills

    print("[Skills] Successfully imported skill modules")
except ImportError as e:
    import traceback
    
    traceback.print_exc()
    print(f"[Skills] Warning: Failed to import some skill modules: {e}")

__all__ = ["basic_skills", "manipulation_skills", "observation_skills"]
