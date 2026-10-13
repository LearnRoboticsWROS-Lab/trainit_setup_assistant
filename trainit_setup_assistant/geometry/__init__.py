"""Build-time geometry: measuring a number off the CAD the cell already ships.

Nothing here runs in a bundle. ``mesh_face`` measures a GRASP FACE on a mesh so the
ranking's denominator is a measurement instead of a bounding-box approximation
(ADR-0018), and ``step_faces`` takes the exact answer from a headless FreeCAD when the
machine happens to have one. Both are pure python + numpy: the measured output is a few
scalars, so a generated bundle keeps no mesh dependency of any kind.
"""

from .mesh_face import (FaceMeasurement, ImplausibleFace, Mesh, OrthoRender,  # noqa: F401
                        PLAUSIBLE_RADIUS_M, VIEWS, coplanar_region, implausible_reason,
                        measure_region, ortho_views, plane_basis, raster, ray_hit,
                        read_mesh, refuse_if_implausible, shade)
from .step_faces import (StepFace, freecad_command, freecad_available,  # noqa: F401
                         match_measurement, read_step_faces)
