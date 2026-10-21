import importlib.util
from pathlib import Path

import pytest

from tidy_arm.planner import OraclePlanner, Plan
from tidy_arm.sim import ITEMS, PLATES, Sim
from tidy_arm.tasks import TASKS

_spec = importlib.util.spec_from_file_location(
    "evaluate", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


@pytest.fixture
def sim():
    s = Sim(seed=1)
    yield s
    s.close()


def test_tasks_are_well_formed(sim):
    assert 20 <= len(TASKS) <= 30
    assert len({t.name for t in TASKS}) == len(TASKS)
    for t in TASKS:
        moves = t.expected(sim)
        assert bool(moves) != t.expect_problem, t.name
        for item, plate in moves:
            assert item in ITEMS and plate in PLATES, t.name


def test_ground_finds_the_pointed_item_and_plate(sim):
    plan = OraclePlanner(sim, [("purple_box", "plate_blue")]).plan(None, "")
    pairs, errors = evaluate.ground(sim, plan)
    assert pairs == [("purple_box", "plate_blue")]
    assert errors[0] < 0.01


def _task(name):
    return next(t for t in TASKS if t.name == name)


def test_judge_classifies_mistakes_without_moving(sim):
    task = _task("basic_red_box_plate_blue")
    expected = task.expected(sim)
    swapped = OraclePlanner(sim, [("purple_box", "plate_blue")]).plan(None, "")
    assert evaluate.judge(sim, task, swapped, expected) == "wrong_item"
    misplaced = OraclePlanner(sim, [("red_box", "plate_yellow")]).plan(None, "")
    assert evaluate.judge(sim, task, misplaced, expected) == "wrong_place"
    assert evaluate.judge(sim, task, Plan(moves=[], problem="x"), expected) == "no_plan"
    two = OraclePlanner(sim, [("red_box", "plate_blue")] * 2).plan(None, "")
    assert evaluate.judge(sim, task, two, expected) == "wrong_count"


def test_judge_infeasible(sim):
    task = _task("infeasible_ball")
    assert evaluate.judge(sim, task, Plan(moves=[], problem="無い"), []) == "ok"
    plan = OraclePlanner(sim, [("red_box", "plate_blue")]).plan(None, "")
    assert evaluate.judge(sim, task, plan, []) == "should_refuse"
