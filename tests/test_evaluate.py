import importlib.util
from pathlib import Path

import pytest

from tidy_arm import tasks
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
        answers, problem_ok = t.accepted(sim)
        assert problem_ok or answers, t.name
        for item, plate in [m for a in answers for m in a]:
            assert item in ITEMS and plate in PLATES, t.name


def test_ground_finds_the_pointed_item_and_plate(sim):
    plan = OraclePlanner(sim, [("purple_box", "plate_blue")]).plan(None, "")
    pairs, errors = evaluate.ground(sim, plan)
    assert pairs == [("purple_box", "plate_blue")]
    assert errors[0] < 0.01


def _task(name):
    return next(t for t in TASKS if t.name == name)


def test_judge_classifies_mistakes_without_moving(sim):
    accepted = _task("basic_red_box_plate_blue").accepted(sim)
    swapped = OraclePlanner(sim, [("purple_box", "plate_blue")]).plan(None, "")
    assert evaluate.judge(sim, swapped, *accepted) == "wrong_item"
    misplaced = OraclePlanner(sim, [("red_box", "plate_yellow")]).plan(None, "")
    assert evaluate.judge(sim, misplaced, *accepted) == "wrong_place"
    assert evaluate.judge(sim, Plan(moves=[], problem="x"), *accepted) == "no_plan"
    two = OraclePlanner(sim, [("red_box", "plate_blue")] * 2).plan(None, "")
    assert evaluate.judge(sim, two, *accepted) == "wrong_count"


def test_judge_infeasible(sim):
    accepted = _task("infeasible_ball").accepted(sim)
    assert evaluate.judge(sim, Plan(moves=[], problem="無い"), *accepted) == "ok"
    plan = OraclePlanner(sim, [("red_box", "plate_blue")]).plan(None, "")
    assert evaluate.judge(sim, plan, *accepted) == "should_refuse"


def test_ambiguous_box_accepts_both_boxes(sim):
    accepted = _task("ambiguous_box").accepted(sim)
    assert evaluate.judge(sim, Plan(moves=[], problem="どれ？"), *accepted) == "ok"
    one = OraclePlanner(sim, [("red_box", "plate_blue")]).plan(None, "")
    assert evaluate.judge(sim, one, *accepted) == "wrong_count"


def test_close_call_accepts_either_or_holding_back():
    best = tasks._best(["a", "b", "c"], {"a": 0.547, "b": 0.548, "c": 0.60}.get)
    assert best == ["a", "b"]
    assert tasks._best(["a", "b"], {"a": 0.50, "b": 0.53}.get) == ["a"]
