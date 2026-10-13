"""TSA has ONE version number: package.xml, setup.py and ``__version__`` agree.

vNext M0C #7. ``__version__`` is what every generated bundle records as its
``generator_version`` (``generation_manifest.yaml``), so it is the number a bundle is traced back
to. It had drifted: package.xml and setup.py said 7.2.0 while ``__version__`` still said 7.1.0 --
and the golden bundle's manifest says 0.1.0, because the release that generated it (v4.3.1) had
never moved it at all. A bundle whose manifest names the wrong generator cannot be reproduced.

Fast and ROS-free.
"""
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

import trainit_setup_assistant
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.model import load_project

PKG = Path(__file__).resolve().parents[1]
EXAMPLE = PKG / 'examples' / 'fr3wml_project.yaml'


def _setup_py_version() -> str:
    m = re.search(r"version\s*=\s*'([^']+)'", (PKG / 'setup.py').read_text())
    assert m, 'no version= in setup.py'
    return m.group(1)


def test_package_xml_setup_py_and_dunder_version_are_one_number():
    xml_version = ET.parse(PKG / 'package.xml').getroot().findtext('version')
    assert xml_version == _setup_py_version() == trainit_setup_assistant.__version__


def test_a_generated_bundle_records_that_number():
    with tempfile.TemporaryDirectory() as tmp:
        Orchestrator().generate(load_project(str(EXAMPLE)), tmp)
        manifest = yaml.safe_load((Path(tmp) / 'generation_manifest.yaml').read_text())
    assert manifest['generator_version'] == trainit_setup_assistant.__version__
