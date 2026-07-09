"""M4 golden test: generating the FR3WML project reproduces the golden fr3wml_app.

Fast + ROS-free (generation is pure). Generates with the GOLDEN package names so the
package dirs line up 1:1, then asserts semantic equivalence to the hand-coded bundle.
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
GOLDEN_ROOT = '/home/fra/fr5_ws/src/fr3wml_digital_twin/fr3wml_app'

GOLDEN_PACKAGES = {
    DESCRIPTION: 'fr3wml_description',
    MOVEIT: 'fr3wml_app_moveit_config',
    APP: 'fr3wml_app',
}


def _generate_with_golden_names(out_dir):
    project = load_project(EXAMPLE)
    project.project_name = 'fr3wml_app'
    project.bundle = BundleSpec(
        description_package=GOLDEN_PACKAGES[DESCRIPTION],
        moveit_config_package=GOLDEN_PACKAGES[MOVEIT],
        app_package=GOLDEN_PACKAGES[APP],
    )
    Orchestrator().generate(project, out_dir)
    return out_dir


@pytest.mark.skipif(not os.path.isdir(GOLDEN_ROOT),
                    reason='golden fr3wml_app bundle not present')
def test_generated_bundle_equals_golden():
    with tempfile.TemporaryDirectory() as tmp:
        _generate_with_golden_names(tmp)
        report = compare_bundle(tmp, GOLDEN_ROOT, GOLDEN_PACKAGES, GOLDEN_PACKAGES)
        assert report.ok, '\n' + report.summary()


@pytest.mark.skipif(not os.path.isdir(GOLDEN_ROOT),
                    reason='golden fr3wml_app bundle not present')
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
