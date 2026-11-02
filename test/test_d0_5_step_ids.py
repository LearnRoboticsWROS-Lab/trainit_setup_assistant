"""D0.5 (decision D18, ADR-0024 amendment, ADR-0030): canonical STEP IDS in the task program.

A step may carry an optional, user-chosen ``id`` (a name, unique within its task, control bodies included). It
names the APPLICATION step in the runtime's execution events; it is never a behavior-tree node identity. The
generated tree changes ONLY where a step has one: that step's node(s) run inside
``<Sequence name="trainit_step:<id>:<kind>">`` (BT.CPP's own ``name`` attribute, the carrier the runner's
observer reads). A project without ids generates byte-identical trees (every existing golden is untouched: the
full suite proves it); a project with ids is pinned here by its own golden, ``test/data/d0_5/task_program.xml``.

The project is the M3.7 composition cell (three tasks: calls, a repeat, a retry inside a fallback, a grasp and
a release) with ids added. ROS-free; the UR5e sample cell needs ur_description (skipped without it, as every
sample test).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

import test_m3_7_composition as C
from trainit_setup_assistant.applications.task_program import STEP_NAME_PREFIX, plan_program, program_tree_xml
from trainit_setup_assistant.composer.ops import step_kind
from trainit_setup_assistant.model import CanonicalProject

GOLDEN = Path(__file__).resolve().parent / 'data' / 'd0_5' / 'task_program.xml'

# (task, path to the step inside the YAML steps, id): a top-level move, a process step, a control step, a
# call inside a repeat body, a move inside a retry inside a fallback, a grasp and a release
IDS = [('cell_cycle', [0], 'start'),
       ('cell_cycle', [1], 'conveyor_on'),
       ('cell_cycle', [2], 'cycles'),
       ('cell_cycle', [2, 'repeat', 'body', 0], 'one_part'),
       ('cell_cycle', [4, 'fallback', 'bodies', 0, 0, 'retry', 'body', 0], 'reach'),
       ('transfer', [0], 'pick'),
       ('transfer', [1], 'place')]


@pytest.fixture(autouse=True)
def _cell(monkeypatch):
    monkeypatch.chdir(C.DATA)
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def _step(data, task, path):
    steps = next(t for t in data['application']['tasks'] if t['name'] == task)['steps']
    node = steps
    for key in path:
        node = node[key]
    return node


def with_ids(ids=IDS):
    def mutate(data):
        for task, path, step_id in ids:
            step = _step(data, task, path)
            items = list(step.items())
            step.clear()
            step.update([('id', step_id)] + items)
    return mutate


def xml_of(project: CanonicalProject) -> str:
    return program_tree_xml(project, plan_program(project))


def unwrap(el: ET.Element) -> ET.Element:
    """The tree with every trainit_step Sequence replaced by its children (in place, recursively)."""
    for parent in list(el.iter()):
        kids = list(parent)
        for i, kid in enumerate(kids):
            if kid.tag == 'Sequence' and (kid.get('name') or '').startswith(STEP_NAME_PREFIX):
                parent.remove(kid)
                for j, grandchild in enumerate(list(kid)):
                    parent.insert(i + j, grandchild)
                return unwrap(el)
    return el


def canon(el: ET.Element):
    return (el.tag, sorted(el.attrib.items()), [canon(k) for k in el])


def test_a_project_without_ids_generates_no_step_wrapper():
    assert STEP_NAME_PREFIX not in xml_of(C.build())


def test_every_identified_step_and_only_it_is_wrapped_with_its_canonical_id_and_kind():
    text = xml_of(C.build(with_ids()))
    root = ET.fromstring(text.split('\n', 1)[1])
    wrappers = {s.get('name') for s in root.iter('Sequence') if (s.get('name') or '').startswith(STEP_NAME_PREFIX)}
    assert wrappers == {'trainit_step:start:move', 'trainit_step:conveyor_on:process', 'trainit_step:cycles:repeat',
                        'trainit_step:one_part:task', 'trainit_step:reach:move', 'trainit_step:pick:grasp',
                        'trainit_step:place:release'}
    trees = {t.get('ID'): t for t in root.iter('BehaviorTree')}
    pick = next(s for s in trees['task_transfer'].iter('Sequence') if s.get('name') == 'trainit_step:pick:grasp')
    assert len(list(pick)) > 1                         # a grasp is several nodes: ONE step, wrapped once
    cycles = next(s for s in trees['task_cell_cycle'].iter('Sequence')
                  if s.get('name') == 'trainit_step:cycles:repeat')
    assert [k.tag for k in cycles] == ['Repeat']       # a control step wraps its control node
    assert 'trainit_step:one_part:task' in {s.get('name') for s in cycles.iter('Sequence')}


def test_removing_the_wrappers_gives_exactly_the_tree_without_ids():
    plain = ET.fromstring(xml_of(C.build()).split('\n', 1)[1])
    wrapped = unwrap(ET.fromstring(xml_of(C.build(with_ids())).split('\n', 1)[1]))
    assert canon(wrapped) == canon(plain)              # the ids change WHERE nodes sit, never which nodes


def test_the_tree_with_ids_is_the_golden_byte_for_byte():
    assert xml_of(C.build(with_ids())) == GOLDEN.read_text(), \
        'the D0.5 step-id golden changed: show the owner the diff before updating it'


def test_tmrs_copy_of_the_golden_is_this_golden_byte_for_byte():
    """ADR-0030 Decision 6: the TSA -> TMR carrier (grammar v1) is pinned on ONE artefact; TMR's observer test runs
    its byte copy. Checked when both repositories are in this workspace."""
    tmr = Path(__file__).resolve().parents[2] / 'trainit_motion_runtime'
    copy = tmr / 'test' / 'data' / 'd0_5' / 'task_program.xml'
    if not copy.is_file():
        pytest.skip(f'{copy} is not in this workspace')
    assert copy.read_bytes() == GOLDEN.read_bytes(), 'update TMR test/data/d0_5/task_program.xml with this golden'


def test_the_same_id_in_two_tasks_is_allowed():
    C.build(with_ids([('cell_cycle', [0], 'begin'), ('go_home', [0], 'begin')]))


@pytest.mark.parametrize('ids, needle', [
    ([('cell_cycle', [0], 'twice'), ('cell_cycle', [4, 'fallback', 'bodies', 0, 0, 'retry', 'body', 0], 'twice')],
     "names two steps 'twice'"),                       # unique within the task, control bodies included
    ([('cell_cycle', [0], '1st')], 'must be a name'),
    ([('cell_cycle', [0], 'true')], 'boolean or null'),
])
def test_a_duplicate_or_malformed_step_id_is_refused(ids, needle):
    with pytest.raises(ValueError, match=needle):
        C.build(with_ids(ids))


def test_the_composer_reads_a_steps_kind_beside_its_id_and_keeps_the_id_on_a_replacement():
    from trainit_setup_assistant.composer.ops import Composer
    assert step_kind({'id': 'start', 'move': {'to': 'home'}}) == 'move'
    composer = Composer(C.build(with_ids()))
    composer.replace_step('cell_cycle', (1,), {'move': {'to': 'home', 'type': 'ptp', 'profile': 'approach'}})
    first = next(t for t in composer.document()['tasks'] if t['name'] == 'cell_cycle')['steps'][0]
    assert first['id'] == 'start' and step_kind(first) == 'move' and first['move']['profile'] == 'approach'
    assert xml_of(composer.project).count('trainit_step:start:move') == 1


def regenerate_golden():  # pragma: no cover -- a deliberate, owner-reviewed update only
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(xml_of(C.build(with_ids())))


if __name__ == '__main__':  # pragma: no cover
    import os
    os.chdir(C.DATA)
    print(yaml.safe_dump({'golden': str(GOLDEN)}))
