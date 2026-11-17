"""TSA 8.8.0 (vNext M4.5d, the owner's brief §23): the camera -> robot frame chain report (livesession/camera_frames).
Pure: what the live graph answered -> what the Perception step shows. The live probe itself needs a ROS graph."""
from trainit_setup_assistant.livesession.camera_frames import (AVAILABLE, MISSING, VALID, evaluate,
                                                               intrinsics_status)

K = [424.0, 0.0, 424.0, 0.0, 424.0, 240.0, 0.0, 0.0, 1.0]
INFO = {'frame_id': 'bin_camera_color_optical_frame', 'k': K, 'width': 848, 'height': 480, 'stamp': '12.5'}
TF = {'translation': [1.95, 1.6, 1.503], 'rotation_xyzw': [0.0, 0.0, 0.0, 1.0]}


def test_a_complete_chain_is_ok():
    r = evaluate(INFO, 'world', 'bin_camera_color_optical_frame', TF)
    assert r.ok and r.tf_status == AVAILABLE and r.calibration == VALID
    text = '\n'.join(r.lines())
    assert 'PERCEPTION FRAME   bin_camera_color_optical_frame' in text and 'TARGET FRAME       world' in text
    assert 'TF STATUS          AVAILABLE at 12.5' in text


def test_a_missing_transform_says_tf2s_reason():
    r = evaluate(INFO, 'world', 'bin_camera_color_optical_frame', None, '"world" passed to lookupTransform does not exist')
    assert not r.ok and r.tf_status == MISSING and 'does not exist' in '\n'.join(r.lines())


def test_no_camera_info_is_reported_not_guessed():
    r = evaluate(None, 'world', 'cam_optical', None)
    assert not r.ok and r.calibration == MISSING and r.problems


def test_a_frame_other_than_the_projects_is_a_problem():
    r = evaluate(INFO, 'world', 'camera_color_optical_frame', TF)
    assert not r.ok and "the project names 'camera_color_optical_frame'" in r.problems[0]


def test_degenerate_intrinsics_are_missing():
    assert intrinsics_status([0.0] * 9, 848, 480)[0] == MISSING
    assert intrinsics_status(K, 848, 480)[0] == VALID
    bad = list(K)
    bad[2] = 900.0                                  # principal point outside the image
    assert intrinsics_status(bad, 848, 480)[0] == MISSING
