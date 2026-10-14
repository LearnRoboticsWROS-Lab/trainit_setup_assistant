"""ADR-0018: the grasp face is MEASURED on the object, not approximated from its box.

What this pins, per Decision:

1. **Decision 1 / 2** — the picker RECOVERS a known number. A synthetic body with a disc
   of a known radius is generated here (no fixture file, no CAD checked in), the face is
   picked by a ray, and the recovered radius is asserted against the polygon's exact
   equal-area radius. This is the regression that matters: everything else in the ADR is
   plumbing around this number.
2. **Decision 2** — three orthographic views, pure numpy, and a CLICK is a triangle: the
   raster's own id buffer answers the pick, so the mouse path and the ray path cannot
   disagree. No VTK, no pyqtgraph, no new import anywhere.
3. **Decision 4** — the unit is explicit and its failure is SILENT, so the same mesh read
   at the wrong scale is REFUSED rather than written: a millimetre file read as metres
   measures a face a thousand times too big and nothing downstream raises.
4. **Decision 6, as ADR-0021 (amendment M2.2) implements it** — TSA READS the face registry
   from its committed snapshot of trainit_perception's vocabulary: the parameters a face takes
   and what becomes computable. A face family TSA has never heard of is configured, refused
   and PINNED correctly after a snapshot refresh (and the owner's classification row) and no
   TSA code, with a snapshot stood in so the test holds whatever ``trainit_perception``
   happens to be installed.
5. **the asymmetry itself** — a hand-written ``rect`` face used to be accepted by the
   runtime and refused by TSA, because ``grasp_face`` was never written. It round-trips
   now, and an unknown face name is refused BY NAME -- on every machine, with or without a
   perception package installed (ADR-0021).

Fast and ROS-free: the geometry is numpy, the generation is pure, the GUI runs offscreen.
"""

import math
import os
import struct
import sys

import numpy as np
import pytest
import yaml

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from trainit_setup_assistant.generator.perception_yaml import (  # noqa: E402
    build_perception_yaml, ranking_param_names, refuse_uncomputable_candidate_rules)
from trainit_setup_assistant.geometry import mesh_face as mf  # noqa: E402
from trainit_setup_assistant.geometry import step_faces as sf  # noqa: E402
from trainit_setup_assistant.model import (BundleSpec, CandidateRule,  # noqa: E402
                                           DetectorSpec, PerceptionSpec, SceneObject,
                                           load_project, project_to_dict)
from trainit_setup_assistant.model import faces as registry  # noqa: E402
from trainit_setup_assistant.model.project import CanonicalProject  # noqa: E402
import identity_cases  # noqa: E402

HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(HERE, os.pardir, 'examples', 'fr3wml_project.yaml')

# The synthetic part: a disc cap of R on a body of 2R, so a wrong flood fill (one that
# swallowed the body's side wall) cannot accidentally produce the right area.
DISC_R = 0.0115                 # 11,5 mm -- the survey's own case
DISC_SEGMENTS = 64
BODY_H = 0.025


@pytest.fixture(scope='module')
def qapp():
    from python_qt_binding.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


# --- the synthetic part, generated here ----------------------------------------------

def _cylinder_tris(radius: float, height: float, segments: int = DISC_SEGMENTS,
                   scale: float = 1.0):
    """A capped cylinder along +Z as (T, 3, 3): a fan cap at z=h, a fan bottom, a wall.

    Written out rather than imported: the point of the test is that the picker needs no
    mesh library, so neither does its fixture.
    """
    ang = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    ring_top = np.stack([radius * np.cos(ang), radius * np.sin(ang),
                         np.full(segments, height)], axis=1)
    ring_bot = ring_top.copy()
    ring_bot[:, 2] = 0.0
    top, bottom = np.array([0.0, 0.0, height]), np.array([0.0, 0.0, 0.0])
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris.append([top, ring_top[i], ring_top[j]])              # the CAP: the face
        tris.append([bottom, ring_bot[j], ring_bot[i]])
        tris.append([ring_bot[i], ring_bot[j], ring_top[j]])      # the wall
        tris.append([ring_bot[i], ring_top[j], ring_top[i]])
    return np.asarray(tris, dtype=np.float64) * scale


def _write_binary_stl(path, tris) -> str:
    """A binary STL, in the layout `read_mesh` parses: 80-byte header, count, 50/triangle."""
    with open(path, 'wb') as f:
        f.write(b'trainit synthetic grasp face'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(tris)))
        for tri in tris:
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            n = n / max(np.linalg.norm(n), 1e-18)
            f.write(struct.pack('<12fH', *n, *tri[0], *tri[1], *tri[2], 0))
    return str(path)


def _exact_polygon_radius(radius: float, segments: int) -> float:
    """The equal-area radius a regular ``segments``-gon inscribed in ``radius`` really has.

    Asserting against the CIRCLE would be asserting against the wrong number: the mesh IS
    a polygon, and the picker measures the mesh. (For 64 segments the polygon holds
    99,92 % of the circle's area, so this is a 0,04 % correction on the radius -- small,
    and exactly the kind of small that a loose tolerance hides.)
    """
    area = 0.5 * segments * radius * radius * math.sin(2 * math.pi / segments)
    return math.sqrt(area / math.pi)


def _synthetic_stl(tmp_path, *, scale: float = 1.0, name: str = 'part.stl') -> str:
    return _write_binary_stl(tmp_path / name,
                             _cylinder_tris(DISC_R, BODY_H, scale=scale))


# --- 1. the number is RECOVERED (Decisions 1 and 2) ----------------------------------

def test_the_picker_recovers_a_known_disc_radius(tmp_path):
    """THE regression. A ray down the cap, a coplanar flood fill, an equal-area radius."""
    path = _synthetic_stl(tmp_path)
    mesh = mf.read_mesh(path, 1.0)
    assert mesh.count == 4 * DISC_SEGMENTS

    hit = mf.ray_hit(mesh.tris, [0.0, 0.0, BODY_H + 0.05], [0.0, 0.0, -1.0])
    assert hit >= 0
    cells = mf.coplanar_region(mesh, hit)
    # exactly the cap: the fan's own triangles, and not one triangle of the wall
    assert len(cells) == DISC_SEGMENTS

    measured = mf.measure_region(mesh, cells)
    expected = _exact_polygon_radius(DISC_R, DISC_SEGMENTS)
    assert measured.radius_m == pytest.approx(expected, abs=1e-6)
    assert measured.radius_m == pytest.approx(DISC_R, rel=2e-3)      # 11,49 vs 11,50 mm
    assert measured.radius_m < DISC_R                               # inscribed, so under
    assert measured.area_m2 == pytest.approx(math.pi * expected ** 2, rel=1e-6)
    assert measured.normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert measured.centroid_m[2] == pytest.approx(BODY_H, abs=1e-9)
    # a disc is the same size in every direction, which is what `DiscFace` collapses
    assert measured.half_a_m == pytest.approx(DISC_R, rel=2e-2)
    assert measured.half_b_m == pytest.approx(measured.half_a_m, rel=5e-2)
    assert mf.implausible_reason(measured) is None


def test_a_ray_that_misses_the_part_reports_nothing(tmp_path):
    mesh = mf.read_mesh(_synthetic_stl(tmp_path), 1.0)
    assert mf.ray_hit(mesh.tris, [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]) == -1


def test_the_flood_fill_does_not_cross_to_the_parallel_face(tmp_path):
    """The cap and the bottom are parallel and 25 mm apart: an angle test alone would merge
    them, which would double the area and halve nothing visibly."""
    mesh = mf.read_mesh(_synthetic_stl(tmp_path), 1.0)
    down = mf.ray_hit(mesh.tris, [0.0, 0.0, -0.05], [0.0, 0.0, 1.0])
    cells = mf.coplanar_region(mesh, down)
    assert len(cells) == DISC_SEGMENTS
    assert mf.measure_region(mesh, cells).centroid_m[2] == pytest.approx(0.0, abs=1e-9)


def test_three_orthographic_views_render_and_a_click_is_a_triangle(tmp_path):
    """Decision 2: no trackball, no GL. The z-buffer resolves occlusion, so the pick is a
    lookup in the id buffer the raster already produced — the mouse cannot disagree with
    the ray."""
    mesh = mf.read_mesh(_synthetic_stl(tmp_path), 1.0)
    views = mf.ortho_views(mesh, 160)
    assert len(views) == 3 == len(mf.VIEWS)
    top = views[2]                                  # looking from +Z: the CAP fills it
    assert top.tid.shape == (160, 160)
    assert (top.tid >= 0).sum() > 0.5 * 160 * 160 * 0.7
    centre = top.triangle_at(80, 80)
    assert centre >= 0
    assert mesh.normals[centre] == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
    assert top.triangle_at(0, 0) == -1               # the corner is background
    assert top.triangle_at(-1, 5) == -1 and top.triangle_at(5, 999) == -1
    # flipped, the SAME view reaches the face on the other side: three views would leave
    # half the part unclickable, and which half depends only on the CAD's orientation
    bottom = mf.raster(mesh, 2, size=160, from_positive=False)
    other = bottom.triangle_at(80, 80)
    assert other >= 0
    assert mesh.normals[other] == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    # the shaded image is what QImage is handed: contiguous uint8 RGB, face highlighted
    cells = mf.coplanar_region(mesh, centre)
    rgb = mf.shade(mesh, top, cells)
    assert rgb.shape == (160, 160, 3) and rgb.dtype == np.uint8
    assert rgb.flags['C_CONTIGUOUS']
    assert rgb[80, 80, 1] > rgb[80, 80, 0]           # green: the marked region


def test_an_obj_file_measures_the_same_face(tmp_path):
    """OBJ in, so a part exported from Blender needs no conversion step."""
    tris = _cylinder_tris(DISC_R, BODY_H)
    lines = []
    for tri in tris:
        for v in tri:
            lines.append(f'v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}')
    for i in range(len(tris)):
        lines.append(f'f {3 * i + 1} {3 * i + 2} {3 * i + 3}')
    path = tmp_path / 'part.obj'
    path.write_text('\n'.join(lines) + '\n')
    mesh = mf.read_mesh(path, 1.0)
    hit = mf.ray_hit(mesh.tris, [0.0, 0.0, BODY_H + 0.05], [0.0, 0.0, -1.0])
    cells = mf.coplanar_region(mesh, hit)
    assert len(cells) == DISC_SEGMENTS
    assert mf.measure_region(mesh, cells).radius_m == pytest.approx(
        _exact_polygon_radius(DISC_R, DISC_SEGMENTS), abs=1e-6)


# --- 2. the unit (Decision 4) ---------------------------------------------------------

def test_a_millimetre_file_read_as_metres_is_REFUSED_not_written(tmp_path):
    """The failure this exists for: a mesh carries no unit, and reading mm as m raises
    NOTHING — the survey's own run reported a 51 m2 face and a 4042 mm radius in silence."""
    path = _synthetic_stl(tmp_path, scale=1000.0, name='in_mm.stl')   # same part, in mm
    as_metres = mf.read_mesh(path, 1.0)
    hit = mf.ray_hit(as_metres.tris, [0.0, 0.0, 1000 * BODY_H + 1.0], [0.0, 0.0, -1.0])
    wrong = mf.measure_region(as_metres, mf.coplanar_region(as_metres, hit))
    reason = mf.implausible_reason(wrong, scale=1.0)
    assert reason and 'unit' in reason.lower() and 'millimetres' in reason
    with pytest.raises(mf.ImplausibleFace):
        mf.refuse_if_implausible(wrong, scale=1.0)

    # the SAME file at the right scale is the same number as the metre part
    right = mf.read_mesh(path, 0.001)
    hit = mf.ray_hit(right.tris, [0.0, 0.0, BODY_H + 0.05], [0.0, 0.0, -1.0])
    good = mf.measure_region(right, mf.coplanar_region(right, hit))
    assert good.radius_m == pytest.approx(_exact_polygon_radius(DISC_R, DISC_SEGMENTS),
                                          abs=1e-6)
    assert mf.implausible_reason(good, scale=0.001) is None


def test_the_refused_band_is_the_RUNTIME_s_own(tmp_path):
    """The picker refuses what the node refuses, because the band is asked of the registry
    and passed in — not kept as a second copy here."""
    tiny = mf.FaceMeasurement(1, 1e-12, 1e-6, 1e-6, 1e-6, (0, 0, 0), (0, 0, 1))
    assert mf.implausible_reason(tiny, band=(0.5, 1.0))
    assert mf.implausible_reason(tiny, band=(1e-9, 1.0)) is None
    lo, hi = registry.size_band()
    assert 0 < lo < hi
    assert mf.implausible_reason(tiny, band=registry.size_band())


def test_an_unreadable_or_unsupported_file_says_what_it_reads(tmp_path):
    with pytest.raises(FileNotFoundError):
        mf.read_mesh(tmp_path / 'nope.stl')
    bad = tmp_path / 'part.dae'
    bad.write_text('<COLLADA/>')
    with pytest.raises(ValueError) as exc:
        mf.read_mesh(bad)
    assert 'STL' in str(exc.value) and 'FreeCAD' in str(exc.value)


# --- 3. STEP stays optional, behind a probe (Decision 3) ------------------------------

def test_step_is_probed_and_never_required(monkeypatch):
    """``freecadcmd`` on PATH or an env override; nothing installed, nothing declared."""
    monkeypatch.delenv(sf.FREECAD_ENV, raising=False)
    monkeypatch.setattr(sf.shutil, 'which', lambda name: None)
    assert sf.freecad_command() is None and sf.freecad_available() is False
    with pytest.raises(RuntimeError) as exc:
        sf.read_step_faces('/nowhere/part.step')
    assert 'OPTIONAL' in str(exc.value) and sf.FREECAD_ENV in str(exc.value)

    monkeypatch.setattr(sf.shutil, 'which',
                        lambda name: '/usr/bin/freecadcmd' if name == 'freecadcmd' else None)
    assert sf.freecad_command() == ['/usr/bin/freecadcmd']


def test_an_appimage_is_told_which_inner_executable_to_run(tmp_path, monkeypatch):
    """An AppImage dispatches to its own ``freecadcmd`` when it is named first, which is how
    a machine with no packaged FreeCAD still gets the exact path."""
    app = tmp_path / 'FreeCAD-1.1.1.AppImage'
    app.write_text('#!/bin/sh\n')
    monkeypatch.setenv(sf.FREECAD_ENV, str(app))
    assert sf.freecad_command() == [str(app), 'freecadcmd']
    monkeypatch.setenv(sf.FREECAD_ENV, f'{app} freecadcmd')
    assert sf.freecad_command() == [str(app), 'freecadcmd']
    monkeypatch.setenv(sf.FREECAD_ENV, str(tmp_path / 'absent'))
    assert sf.freecad_command() is None


def test_an_exact_brep_face_matches_the_tessellation_it_belongs_to():
    """So a click on a STEP file's tessellation can report the EXACT area — and refuses to
    guess when nothing agrees, because a wrong match publishes another face's radius."""
    exact = sf.StepFace(index=6, kind='Plane', area_m2=1.37885e-3, radius_m=0.02095,
                        half_a_m=0.02095, half_b_m=0.02094, centre_m=(0.0, 0.0, 0.01),
                        axis=(0.0, 0.0, 1.0))
    close = mf.FaceMeasurement(120, 1.370e-3, 0.02089, 0.0209, 0.0209,
                               (0.0001, 0.0, 0.0101), (0.0, 0.0, 1.0))
    assert sf.match_measurement(close, [exact]) is exact
    assert sf.match_measurement(close, [exact]).as_measurement().exact is True
    assert sf.match_measurement(close, [exact]).as_measurement().radius_m == 0.02095
    tilted = mf.FaceMeasurement(120, 1.370e-3, 0.02089, 0.0209, 0.0209,
                                (0.0, 0.0, 0.0101), (0.0, 0.7, 0.714))
    assert sf.match_measurement(tilted, [exact]) is None          # wrong normal
    far = mf.FaceMeasurement(120, 1.370e-3, 0.02089, 0.0209, 0.0209,
                             (0.0, 0.0, 0.05), (0.0, 0.0, 1.0))
    assert sf.match_measurement(far, [exact]) is None             # wrong place
    small = mf.FaceMeasurement(120, 2.0e-4, 0.008, 0.008, 0.008,
                               (0.0, 0.0, 0.0101), (0.0, 0.0, 1.0))
    assert sf.match_measurement(small, [exact]) is None           # wrong size


# --- 4. the registry is READ from TSA's snapshot (Decision 6; ADR-0021, M2.2) ----------

@pytest.fixture(autouse=True)
def _registry_cache():
    """Re-read the face registry after every test.

    The registry answer is CACHED (a wizard that asks once per keystroke must not pay for
    re-reading it), so a test that stands in a snapshot has to put it back — and doing it here
    means no test can forget.
    """
    yield
    registry.face_registry(refresh=True)


def _stub_registry(monkeypatch, *, faces):
    """Stand in a TP vocabulary snapshot whose faces are ``faces``.

    A STUB on purpose: the assertions below must hold whatever the committed snapshot and
    whatever perception package (none included) the machine has. It is TSA's committed snapshot
    with the faces replaced and their parameters dated -- held to the same schema the real one
    is (``parse_tp_registry``), so what is tested is that a new face family needs a snapshot
    refresh and no TSA code.
    """
    from trainit_setup_assistant import catalog
    record = catalog.tp_registry()
    record['faces'] = {name: {'params': list(row['params']),
                              'enables_terms': list(row['enables_terms'])}
                       for name, row in faces.items()}
    for row in faces.values():
        for param in row['params']:
            record['param_since'].setdefault(param, '0.9.0')
    record = catalog.parse_tp_registry(catalog.tp_registry_text(record).encode(), 'the stub')
    monkeypatch.setattr(catalog, '_tp_snapshot', lambda: record)
    registry.face_registry(refresh=True)


BLOB = {'params': ('expected_blob_radius_m', 'expected_blob_depth_m'),
        'enables_terms': ('visible_fraction',)}
DISC = {'params': ('expected_radius_m',), 'enables_terms': ('visible_fraction',)}


def test_tsa_learns_a_face_family_it_has_never_heard_of(monkeypatch):
    """Decision 6: the params a face takes and what becomes computable come from the registry
    -- through TSA's snapshot of it (ADR-0021) -- so a new face family needs no TSA code: a
    refresh, after the owner's classification row. ``blob`` exists nowhere in TSA's code. What a
    face measures (its extent terms) is not part of what TSA knows at all (ADR-0021, D9)."""
    _stub_registry(monkeypatch, faces={'disc': DISC, 'blob': BLOB})
    assert registry.available_faces() == ('blob', 'disc')
    assert registry.registry_source().startswith("TSA's snapshot of trainit_perception")
    assert registry.face_params('blob') == BLOB['params']
    assert not hasattr(registry, 'extent_term_keys')
    assert registry.all_face_params() == ('expected_blob_depth_m',
                                          'expected_blob_radius_m', 'expected_radius_m')
    # ... and every one of them is a ranking parameter the version pin covers
    assert set(BLOB['params']) <= set(ranking_param_names())
    assert 'grasp_face' in ranking_param_names()
    assert {registry.param_floors()[k] for k in BLOB['params']} == {'0.9.0'}
    # what becomes computable is the face's answer too
    assert registry.face_term_requirement(
        'visible_fraction', {'grasp_face': 'blob'}) == BLOB['params']
    assert registry.face_term_requirement('clearance', {'grasp_face': 'blob'}) == ()


def test_a_measured_blob_face_reaches_perception_yaml_whole(monkeypatch):
    """The asymmetry Decision 6 closes, in one assertion: TSA writes ``grasp_face`` AND the
    parameters that face is sized by, for a family it knows nothing about."""
    _stub_registry(monkeypatch, faces={'disc': DISC, 'blob': BLOB})
    p = _project(CandidateRule.SCORE_X_VISIBLE, face='blob',
                 measured={'expected_blob_radius_m': 0.021,
                           'expected_blob_depth_m': 0.004})
    params = _entry(p)['params']
    assert params['grasp_face'] == 'blob'
    assert params['expected_blob_radius_m'] == 0.021
    assert params['expected_blob_depth_m'] == 0.004
    assert 'expected_radius_m' not in params         # the disc's parameter is not a face's
    refuse_uncomputable_candidate_rules(p)           # computable: the face is sized


def test_a_blob_face_with_a_missing_parameter_is_refused_by_NAME(monkeypatch):
    """Half-configured is not configured: ``visible_fraction`` divides by the face, so a
    face missing one of ITS numbers must be as loud as an unset tool radius."""
    _stub_registry(monkeypatch, faces={'disc': DISC, 'blob': BLOB})
    p = _project(CandidateRule.SCORE_X_VISIBLE, face='blob',
                 measured={'expected_blob_radius_m': 0.021})
    with pytest.raises(ValueError) as exc:
        refuse_uncomputable_candidate_rules(p)
    message = str(exc.value)
    assert 'visible_fraction' in message
    assert 'expected_blob_depth_m' in message and 'blob' in message


def test_an_unknown_face_is_refused_when_the_registry_CAN_answer(monkeypatch):
    _stub_registry(monkeypatch, faces={'disc': DISC})
    p = _project(CandidateRule.SCORE_X_VISIBLE, face='blob',
                 measured={'expected_blob_radius_m': 0.021})
    with pytest.raises(ValueError) as exc:
        refuse_uncomputable_candidate_rules(p)
    assert 'blob' in str(exc.value) and 'disc' in str(exc.value)


def test_an_unknown_face_is_refused_with_no_perception_package_installed(monkeypatch):
    """ADR-0021 (amendment M2.2) supersedes the old answer: before, a face TSA could not look up
    on a machine without trainit_perception was NOT refused -- so the same project generated on
    one machine and was refused on another. The snapshot always answers: the refusal is the
    same everywhere, and it is the one the node would give at start-up."""
    for name in ('trainit_perception', 'trainit_perception.ranking',
                 'trainit_perception.ranking.faces', 'trainit_perception.vocabulary'):
        monkeypatch.setitem(sys.modules, name, None)               # import raises
    registry.face_registry(refresh=True)
    assert registry.registry_source().startswith("TSA's snapshot of trainit_perception")
    from trainit_setup_assistant import catalog
    assert registry.available_faces() == tuple(sorted(catalog.tp_registry()['faces']))
    p = _project(CandidateRule.SCORE_X_VISIBLE, face='blob',
                 measured={'expected_blob_radius_m': 0.021})
    with pytest.raises(ValueError) as exc:
        refuse_uncomputable_candidate_rules(p)
    assert 'grasp_face "blob" is not a face trainit_perception registers' in str(exc.value)


# --- 5. the measurement replaces the approximation, and changes no contract ----------

def _project(rule=None, *, face: str = '', measured=None, target='mesh',
             params=None, method='color_mask'):
    p = load_project(EXAMPLE)
    p.bundle = BundleSpec.from_prefix('cell')
    p.perception = p.perception or PerceptionSpec()
    p.perception.detectors = [DetectorSpec(name='capsules', method=method,
                                           params=dict(params or {}),
                                           candidate_rule=rule)]
    if target is not None:
        p.scene.objects = [o for o in p.scene.objects if not o.is_grasp_target()] + [
            SceneObject(id='capsule', shape=target, dims=[0.0419, 0.0592, 0.0188],
                        category='dynamic', grasp_target=True,
                        grasp_face=face, grasp_face_params=dict(measured or {}),
                        grasp_face_source=('measured in a test' if measured else ''))]
    return p


def _entry(project, name='capsules'):
    return yaml.safe_load(build_perception_yaml(project))['detectors'][name]


def test_the_measurement_replaces_the_bounding_box_number():
    """The ADR's own numbers: the capsule's AABB gives 28,10 mm where the true face is
    20,95 mm — +34 % on the radius and +80 % on the area visible_fraction divides by."""
    approximated = _entry(_project(CandidateRule.SCORE_X_VISIBLE))['params']
    assert approximated['expected_radius_m'] == pytest.approx(0.02810, abs=1e-5)
    assert 'grasp_face' not in approximated          # the default face writes no token

    measured = _entry(_project(CandidateRule.SCORE_X_VISIBLE, face='disc',
                               measured={'expected_radius_m': 0.02095}))['params']
    assert measured['expected_radius_m'] == 0.02095
    assert 'grasp_face' not in measured              # still the default face
    area_ratio = (approximated['expected_radius_m'] / measured['expected_radius_m']) ** 2
    assert area_ratio == pytest.approx(1.80, abs=0.02)


def test_the_measured_face_round_trips_through_project_yaml():
    p = _project(CandidateRule.SCORE_X_VISIBLE, face='disc',
                 measured={'expected_radius_m': 0.02095})
    back = CanonicalProject.parse_obj(project_to_dict(p))
    obj = back.scene.first_grasp_target()
    assert obj.grasp_face_params == {'expected_radius_m': 0.02095}
    assert obj.has_measured_face() and obj.face_name() == 'disc'
    assert obj.grasp_face_source == 'measured in a test'
    assert back.scene.grasp_face() == ('disc', {'expected_radius_m': 0.02095})
    assert back.scene.grasp_face_radius_m() == 0.02095


def test_a_hand_edited_params_value_still_wins_over_the_measurement():
    """``params`` is a verbatim runtime profile a configurator may edit by hand, and an
    upsert must not strip it — the measurement is one more derivation, not an override."""
    p = _project(CandidateRule.SCORE_X_VISIBLE, face='disc',
                 measured={'expected_radius_m': 0.02095},
                 params={'expected_radius_m': 0.009})
    assert _entry(p)['params']['expected_radius_m'] == 0.009


def test_an_unmeasured_project_writes_exactly_what_it_wrote_before():
    """The byte-stability promise, asserted where it is cheap: no face was measured, so the
    file carries the derived radius and NO new key."""
    params = _entry(_project(CandidateRule.SCORE_X_VISIBLE))['params']
    assert set(params) >= {'candidate_rule', 'expected_radius_m'}
    assert 'grasp_face' not in params
    unranked = _entry(_project(None))['params']
    assert not ({'candidate_rule', 'grasp_face', 'expected_radius_m'} & set(unranked))


def test_clearing_the_measurement_returns_to_the_approximation():
    p = _project(CandidateRule.SCORE_X_VISIBLE, face='disc',
                 measured={'expected_radius_m': 0.02095})
    obj = p.scene.first_grasp_target()
    obj.grasp_face, obj.grasp_face_params, obj.grasp_face_source = '', {}, ''
    assert _entry(p)['params']['expected_radius_m'] == pytest.approx(0.02810, abs=1e-5)


# --- 6. the GUI, headless --------------------------------------------------------------

def test_the_controller_writes_the_measurement_onto_the_grasp_target():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(_project(None))
    info = ctrl.grasp_face_target()
    assert info['id'] == 'capsule' and info['face'] == 'disc'
    assert info['derived_radius_m'] == pytest.approx(0.02810, abs=1e-5)
    assert 'bounding box' in ctrl.grasp_face_summary()

    ctrl.set_grasp_face('disc', {'expected_radius_m': 0.02095}, 'capsule_visual.step')
    assert ctrl.project.scene.grasp_face_radius_m() == 0.02095
    assert 'MEASURED' in ctrl.grasp_face_summary()
    assert 'capsule_visual.step' in ctrl.grasp_face_summary()

    ctrl.set_grasp_face('disc', {})                     # cleared
    assert ctrl.project.scene.grasp_face_radius_m() == pytest.approx(0.02810, abs=1e-5)


def test_measuring_a_face_cannot_be_asked_of_a_cell_with_no_grasp_target():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(_project(None, target=None))
    assert ctrl.grasp_face_target() is None
    assert 'no grasp target' in ctrl.grasp_face_summary()
    with pytest.raises(RuntimeError):
        ctrl.set_grasp_face('disc', {'expected_radius_m': 0.02})


def test_the_picker_dialog_measures_a_click_offscreen(qapp, tmp_path):
    """The whole dialog, driven headless: load a file, click a view, read the answer back.

    This is the one place the Qt skin is exercised — it is the first widget in the wizard
    that paints and handles a mouse, so it is also the first that can be driven like this.
    """
    from trainit_setup_assistant.gui.face_picker import GraspFacePicker
    path = _synthetic_stl(tmp_path)
    dialog = GraspFacePicker(mesh_path=path, scale=1.0)
    dialog.show()
    dialog._tick()                                   # the polled cache does the loading
    assert dialog._mesh is not None and len(dialog._renders) == 3
    top = dialog.views[2]
    assert top.pixmap() is not None and not top.pixmap().isNull()

    # a widget click, mapped through the SAME aspect-preserving scaling Qt applied
    pixel = top.image_pixel(top.width() // 2, top.height() // 2)
    assert pixel is not None
    dialog._on_pick(top._render.triangle_at(*pixel))
    face, params, source = dialog.result_face()
    assert face == 'disc'
    assert params['expected_radius_m'] == pytest.approx(
        _exact_polygon_radius(DISC_R, DISC_SEGMENTS), abs=1e-5)
    assert os.path.basename(path) in source and 'triangles' in source
    assert dialog.buttons.button(dialog.buttons.Ok).isEnabled()


def test_the_dialog_refuses_a_face_measured_at_the_wrong_unit(qapp, tmp_path):
    from trainit_setup_assistant.gui.face_picker import GraspFacePicker
    path = _synthetic_stl(tmp_path, scale=1000.0, name='in_mm.stl')
    dialog = GraspFacePicker(mesh_path=path, scale=1.0)      # wrong: the file is in mm
    dialog.show()
    dialog._tick()
    top = dialog.views[2]
    pixel = top.image_pixel(top.width() // 2, top.height() // 2)
    dialog._on_pick(top._render.triangle_at(*pixel))
    assert 'unit' in dialog.status.text().lower()
    assert not dialog.buttons.button(dialog.buttons.Ok).isEnabled()
    assert dialog.result_face()[1] == {}                     # nothing to write

    dialog._select_scale(0.001)                              # the user fixes the unit
    dialog._request_load()
    dialog._tick()
    top = dialog.views[2]
    dialog._on_pick(top._render.triangle_at(
        *top.image_pixel(top.width() // 2, top.height() // 2)))
    assert dialog.result_face()[1]['expected_radius_m'] == pytest.approx(
        _exact_polygon_radius(DISC_R, DISC_SEGMENTS), abs=1e-5)


def test_the_dialog_renders_the_FACE_REGISTRY_s_parameters(monkeypatch, qapp, tmp_path):
    """One spin box per parameter the configured face declares — so a face family added to
    the registry is configurable here without a line of the dialog changing."""
    _stub_registry(monkeypatch, faces={'disc': DISC, 'blob': BLOB})
    from trainit_setup_assistant.gui.face_picker import GraspFacePicker
    dialog = GraspFacePicker(mesh_path=_synthetic_stl(tmp_path), scale=1.0, face='blob')
    dialog.show()
    dialog._tick()
    assert set(dialog._param_fields) == set(BLOB['params'])
    top = dialog.views[2]
    dialog._on_pick(top._render.triangle_at(
        *top.image_pixel(top.width() // 2, top.height() // 2)))
    face, params, _ = dialog.result_face()
    assert face == 'blob'
    # the convention fills the radius it recognises and SAYS it cannot fill the other
    assert params['expected_blob_radius_m'] == pytest.approx(
        _exact_polygon_radius(DISC_R, DISC_SEGMENTS), abs=1e-5)
    assert 'expected_blob_depth_m' not in params
    assert 'expected_blob_depth_m' in dialog.status.text()


def test_the_perception_page_offers_the_picker_and_keeps_what_it_measured(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import PerceptionPage
    ctrl = AssistantController(_project(None))
    page = PerceptionPage(ctrl)
    page.initializePage()
    assert page.face_btn.isEnabled()
    assert 'bounding box' in page.face_summary.text()
    ctrl.set_grasp_face('disc', {'expected_radius_m': 0.02095}, 'capsule_visual.step')
    page._refresh_face()
    assert 'MEASURED' in page.face_summary.text()
    # and the measured face UNLOCKS the rule that divides by it
    offered, preferred = ctrl.candidate_rules_for()
    assert CandidateRule.SCORE_X_VISIBLE in offered and preferred.value == 'score_x_visible'


def test_the_picker_button_is_dead_without_a_grasp_target(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import PerceptionPage
    page = PerceptionPage(AssistantController(_project(None, target=None)))
    page.initializePage()
    assert not page.face_btn.isEnabled()
    assert 'no grasp target' in page.face_summary.text()


# --- 7. a real part's CAD ------------------------------------------------------------
# The check of ADR-0018's numbers against a REAL part's CAD (its STL and STEP, with a
# headless FreeCAD) lives in the repository that owns that CAD: a framework test may not
# point at a customer's files. Moved there in vNext M0B; the synthetic part above is the
# regression that matters here.


# --- 8. all the way into a generated bundle -------------------------------------------

def test_a_measured_face_reaches_a_generated_bundle(tmp_path):
    """The measurement is configuration like any other: GUI -> project -> perception.yaml,
    and the version pin moves with it."""
    from trainit_setup_assistant.generator import Orchestrator
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(_project(CandidateRule.SCORE_X_VISIBLE))
    ctrl.set_grasp_face('disc', {'expected_radius_m': 0.02095}, 'capsule_visual.step face[6]')
    out = tmp_path / 'bundle'
    identity_cases.require('p2_example')
    Orchestrator().generate(ctrl.project, str(out))
    written = yaml.safe_load((out / 'cell_app' / 'config' / 'perception.yaml').read_text())
    params = written['detectors']['capsules']['params']
    assert params['expected_radius_m'] == 0.02095
    assert params['candidate_rule'] == 'score_x_visible'
    assert 'grasp_face' not in params                 # the default face writes no token
    assert ('<exec_depend version_gte="0.3.0">trainit_perception</exec_depend>'
            in (out / 'cell_app' / 'package.xml').read_text())
