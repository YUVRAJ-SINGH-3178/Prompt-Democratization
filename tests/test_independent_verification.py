import random

from app.linter.core import ATOMS, evaluate
from app.sample_prompts import SAMPLES
from app.verification import direct_energy, independent_metrics


def test_independent_energy_matches_direct_enumeration_for_200_random_sets():
    rng = random.Random(20261117)
    for _ in range(200):
        atoms = [atom for atom in ATOMS if rng.choice((True, False))]
        production = evaluate("", atoms=atoms)
        assert production.energy == direct_energy(atoms)


def test_independent_slot_scores_match_sample_library():
    for sample in SAMPLES:
        atoms = evaluate(sample["text"]).atoms
        reference = independent_metrics(sample["text"], atoms)
        production = evaluate(sample["text"])
        assert production.energy == reference["energy"]
        assert production.slot_scores == reference["slots"]
        assert round(production.completeness, 3) == round(reference["completeness"], 3)
