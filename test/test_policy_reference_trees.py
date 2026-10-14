"""Reference trees for the learned-policy path: Pure and Hybrid on the reference cell (vNext M0D T1).

``test_policy_application.py`` pins the policy step token by token. This pins what a policy step
does to a WHOLE bundle: the reference cell's project (the golden's) plus one policy step before
``pre_place``, generated in full and hashed, for PURE and for HYBRID. Two facts:

* the policy path is DETERMINISTIC and its bytes are pinned (any change to them is a deliberate,
  explained change of the constants below);
* a policy step is ADDITIVE: against the same project with no policy, it adds exactly the card,
  the exported model files and the runtime launch, and changes exactly the tree, the bring-up and
  the CMakeLists that installs the policies -- nothing else in the bundle moves.

The policy card is a FIXTURE written here, not trainit_lab's live card, which changes with
training (its attach_radius moved from 0.02 to 0.015 after the owner's policy bundle was made).
generation_manifest.yaml is left out of the hash: it records PROVENANCE -- the TSA release and
the identity of its catalog snapshot (vNext M2.3), which every release and refresh changes; the
golden gate and test_bundle_provenance.py check it (before M2.3 it also recorded the card's
temporary source path; a COPY's source is ``external:policy_card.yaml`` now).

Measured 2026-10-06 with the owner's own policy bundle (fr3wml_suction_policy_test4_bundle,
untracked, not used here): it regenerates byte-identical except that copied card and the
manifest's generator_version.

Re-pinned once, at the owner-approved M2.4 re-bless (2026-10-08): TSA 7.6.0 writes the runtime
requirement the M2.3 pins predate -- the TMR floor in both package.xml files, and in the app's
the learned-policy runtime a policy step needs. Before the re-pin, each tree with exactly those
lines put back hashed to its M2.3 constant (pure c308776d..., hybrid 6bc9462d...); no other file
differed. That a policy step also changes the app's package.xml -- by the policy runtime's
dependency and nothing else -- is the M2.4 contract and is asserted as such.

Re-pinned again at the owner-approved vNext G0 re-bless (2026-10-14): TSA 7.8.1's trainit_config
package.xml declares the controller manager its copied moveit_controllers.yaml names. With those
three lines removed each tree hashes to its previous constant (pure 83ee32cd..., hybrid
76d30cda...); no other file differed.

Fast and ROS-free.
"""
import hashlib
import re
import tempfile
import textwrap
from pathlib import Path

import pytest

import ws_paths
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.model import BundleSpec, PolicyStep, load_project
from trainit_setup_assistant.model.enums import PolicyMode

GOLDEN_PROJECT = ws_paths.GOLDEN_BUNDLE / 'project.yaml'
pytestmark = pytest.mark.skipif(not GOLDEN_PROJECT.is_file(), reason='golden bundle not present')

_CARD = textwrap.dedent("""
    policy_card:
      name: fr5_suction_pickplace
      trained_for: "reach + grasp + carry to pre-place with the vacuum active"
      files: {jit: policy.pt, onnx: policy.onnx}
      end_state:
        tcp_pose_base: [0.4267, -0.278, 0.1589, -0.006, -0.7006, 0.7134, 0.0117]
        object_attached: true
        suction_active: true
""")

# sha256 over "<sha256>  <posix relative path>\n" of every generated file but the manifest
REFERENCE_TREE_SHA256 = {
    PolicyMode.PURE: 'a2a4397172f46cc04ecdd363fb1ae74020805b62cba2d47a76eadd0371b69f05',
    PolicyMode.HYBRID: 'f4de26fdd7ac1ad676a111f43638eb3df64fe967298a766d9438e13bf48c4d70',
}
ADDED = {'cell_app/launch/policy_runtime.launch.py',
         'cell_app/policies/grasp2preplace/policy.onnx',
         'cell_app/policies/grasp2preplace/policy.pt',
         'cell_app/policies/grasp2preplace/policy_card.yaml'}
CHANGED = {'cell_app/CMakeLists.txt', 'cell_app/bt_trees/pick_place.xml',
           'cell_app/launch/bringup.launch.py', 'cell_app/package.xml'}


def _tree(mode=None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        card = tmp / 'card'
        card.mkdir()
        (card / 'policy_card.yaml').write_text(_CARD)
        (card / 'policy.onnx').write_bytes(b'ONNX')
        (card / 'policy.pt').write_bytes(b'JIT')
        p = load_project(str(GOLDEN_PROJECT))
        p.bundle = BundleSpec.from_prefix('cell')
        if mode is not None:
            p.application.policies = [PolicyStep(
                name='grasp2preplace', before_waypoint='pre_place', mode=mode,
                card=str(card / 'policy_card.yaml'), require_attached=True,
                attached_topic='/suction/attached')]
        Orchestrator().generate(p, tmp / 'out')
        out = tmp / 'out'
        return {f.relative_to(out).as_posix(): f.read_bytes() for f in sorted(out.rglob('*'))
                if f.is_file() and f.name != 'generation_manifest.yaml'}


def _sha(files: dict) -> str:
    return hashlib.sha256(''.join(f'{hashlib.sha256(b).hexdigest()}  {rel}\n'
                                  for rel, b in sorted(files.items())).encode()).hexdigest()


@pytest.mark.parametrize('mode', [PolicyMode.PURE, PolicyMode.HYBRID])
def test_the_policy_reference_tree_is_pinned_and_deterministic(mode):
    first = _sha(_tree(mode))
    assert first == _sha(_tree(mode))
    assert first == REFERENCE_TREE_SHA256[mode]


@pytest.mark.parametrize('mode', [PolicyMode.PURE, PolicyMode.HYBRID])
def test_a_policy_step_is_additive(mode):
    with_policy, without = _tree(mode), _tree(None)
    assert set(with_policy) - set(without) == ADDED
    assert set(without) - set(with_policy) == set()
    assert {k for k in without if with_policy[k] != without[k]} == CHANGED
    # the app's package.xml gains the policy runtime's dependency -- one comment and the
    # exec_depend, right after the TP line -- and nothing else (vNext M2.4)
    before = without['cell_app/package.xml'].decode().splitlines()
    after = with_policy['cell_app/package.xml'].decode().splitlines()
    at = before.index('  <exec_depend>trainit_perception</exec_depend>') + 1
    added = after[at:len(after) - len(before) + at]
    assert after[:at] + after[at + len(added):] == before
    assert re.fullmatch(r'  <!-- the learned-policy runtime \(ADR-0005\).*', added[0]), added
    assert added[-1] == '  <exec_depend>trainit_policy_runtime</exec_depend>', added
    assert all('--' not in line.strip()[4:-3] for line in added if '<!--' in line)
