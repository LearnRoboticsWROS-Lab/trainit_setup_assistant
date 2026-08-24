"""Golden test: TSA reproduces the bundle it published.

The golden used to be ``fr3wml_app``, a HAND-WRITTEN bundle that the generator was
asked to match. That bundle was removed on 2026-08-24, and the golden is now
``fr3wml_suction_tsa_32_bundle`` -- a real TSA output, validated end to end in Isaac
Sim, which ships its own ``project.yaml``.

That makes this a stronger test than it was. Before, it asked "does the generator
match something a human wrote?". Now it asks "does the generator still produce the
bundle it is on record as having produced?" -- so any change to a template, an
emitter or a default that would silently alter a published bundle fails here.

Fast and ROS-free: generation is pure.
"""

import os
import tempfile

import pytest

from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.model import BundleSpec, load_project
from trainit_setup_assistant.verify import compare_bundle
from trainit_setup_assistant.verify.equivalence import APP, DESCRIPTION, MOVEIT

HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(HERE, os.pardir, 'examples', 'fr3wml_project.yaml')
GOLDEN_ROOT = ('/home/fra/fr5_ws/src/fr3wml_digital_twin/'
               'fr3wml_suction_tsa_32_bundle')
GOLDEN_PROJECT = os.path.join(GOLDEN_ROOT, 'project.yaml')

GOLDEN_PACKAGES = {
    DESCRIPTION: 'fr3wml_suction_tsa_32_description',
    MOVEIT: 'fr3wml_suction_tsa_32_trainit_config',
    APP: 'fr3wml_suction_tsa_32_app',
}


def _generate_with_golden_names(out_dir):
    # The bundle ships the exact project it was generated from, so no renaming is
    # needed: regenerate it verbatim and compare against what was published.
    project = load_project(GOLDEN_PROJECT)
    project.bundle = BundleSpec(
        description_package=GOLDEN_PACKAGES[DESCRIPTION],
        moveit_config_package=GOLDEN_PACKAGES[MOVEIT],
        app_package=GOLDEN_PACKAGES[APP],
    )
    Orchestrator().generate(project, out_dir)
    return out_dir


@pytest.mark.skipif(not os.path.isdir(GOLDEN_ROOT),
                    reason='golden TSA v3.2 bundle not present')
def test_generated_bundle_equals_golden():
    with tempfile.TemporaryDirectory() as tmp:
        _generate_with_golden_names(tmp)
        report = compare_bundle(tmp, GOLDEN_ROOT, GOLDEN_PACKAGES, GOLDEN_PACKAGES)
        assert report.ok, '\n' + report.summary()


@pytest.mark.skipif(not os.path.isdir(GOLDEN_ROOT),
                    reason='golden TSA v3.2 bundle not present')
def test_bt_params_and_tree_semantically_equal():
    """Focused check on the application layer (the generator's core value)."""
    with tempfile.TemporaryDirectory() as tmp:
        _generate_with_golden_names(tmp)
        report = compare_bundle(tmp, GOLDEN_ROOT, GOLDEN_PACKAGES, GOLDEN_PACKAGES)
        app_checks = [c for c in report.checks
                      if c[0].startswith('app/') and c[1] in ('yaml', 'tree')]
        assert app_checks, 'no app-layer checks ran'
        failed = [c for c in app_checks if not c[2]]
        assert not failed, '\n' + '\n'.join(f'{c[0]}: {c[3]}' for c in failed)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
