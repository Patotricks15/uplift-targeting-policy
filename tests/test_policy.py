import numpy as np

from uplift_policy.metrics import qini_coefficient, qini_curve
from uplift_policy.policy import apply_cutoff, choose_cutoff, cumulative_net_gain, evaluate_policy, top_fraction_mask


# Targeting order by score is rows 0, 1, 2, 3: (y=1,t=1), (y=0,t=0), (y=0,t=1), (y=1,t=0).
Y = [1, 0, 0, 1]
T = [1, 0, 1, 0]
SCORE = [4, 3, 2, 1]


def test_budget_cut_treats_exactly_the_top_share():
    score = [0.1, 0.9, 0.5, 0.7, 0.3, 0.2, 0.8, 0.4, 0.6, 0.0]

    mask = top_fraction_mask(score, 0.3)

    assert mask.sum() == 3
    assert set(np.flatnonzero(mask)) == {1, 6, 3}  # scores 0.9, 0.8, 0.7
    assert top_fraction_mask(score, 0).sum() == 0


def test_net_gain_per_k_matches_hand_calculation():
    table = cumulative_net_gain(Y, T, SCORE, cost=0.25)

    # k=1 has no control; k=2: 2*(1-0) - 2*0.25 = 1.5; k=3: 3*(0.5-0) - 3*0.25 = 0.75; k=4: 0 - 4*0.25 = -1
    assert table["net_gain"].iloc[0] == -np.inf
    np.testing.assert_allclose(table["net_gain"].iloc[1:], [1.5, 0.75, -1.0])


def test_cutoff_is_chosen_from_given_data_and_treats_nobody_when_unprofitable():
    best = choose_cutoff(Y, T, SCORE, cost=0.25)
    assert best["fraction"] == 0.5
    assert best["cutoff_score"] == 3
    np.testing.assert_allclose(best["net_gain"], 1.5)

    none = choose_cutoff(Y, T, SCORE, cost=1.5)
    assert none["fraction"] == 0.0
    assert not apply_cutoff(SCORE, none["cutoff_score"]).any()

    mask = apply_cutoff(SCORE, best["cutoff_score"])
    np.testing.assert_array_equal(mask, [True, True, False, False])
    np.testing.assert_allclose(evaluate_policy(Y, T, mask, cost=0.25)["net_gain"], 1.5)

def test_qini_curve_and_coefficient_match_hand_calculation():
    # Cumulative gain along the score order: 0, 1, 1, 1, 0 at x = 0, 0.25, 0.5, 0.75, 1.
    x, gain = qini_curve(Y, T, SCORE)
    np.testing.assert_allclose(x, [0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(gain, [0, 1, 1, 1, 0])
    # Trapezoid area of the curve minus the random line (random line ends at 0 here): 0.75.
    assert np.isclose(qini_coefficient(Y, T, SCORE), 0.75)
