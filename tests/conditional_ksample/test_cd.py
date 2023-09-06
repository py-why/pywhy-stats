import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from sklearn.ensemble import RandomForestClassifier

from pywhy_stats import bregman, kcd

seed = 12345
rng = np.random.default_rng(seed)
# number of samples to use in generating test dataset; the lower the faster
n_samples = 150


def _single_env_scm(n_samples=200, offset=0.0, seed=None):
    # We construct a SCM where X1 -> Y <- X and Y -> Z
    # so X1 is independent from X, but conditionally dependent
    # given Y or Z
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n_samples, 1)) + offset
    X1 = rng.standard_normal((n_samples, 1)) + offset
    Y = X + X1 + 0.05 * rng.standard_normal((n_samples, 1))
    Z = Y + 0.05 * rng.standard_normal((n_samples, 1))

    # create input for the CD test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])

    # assign groups randomly
    df["group"] = rng.choice([0, 1], size=len(df))
    return df


def _multi_env_scm(n_samples=100, offset=1.5, seed=None):
    df = _single_env_scm(n_samples=n_samples, seed=seed)
    df["group"] = 0

    new_df = _single_env_scm(n_samples=n_samples, offset=offset, seed=seed)
    new_df["group"] = 1
    df = pd.concat((df, new_df), axis=0)
    return df


@pytest.mark.parametrize(
    "cd_func",
    [
        kcd,
        bregman,
    ],
)
def test_cd_tests_error(cd_func):
    x = "x"
    y = "y"
    sample_df = _single_env_scm(n_samples=10, seed=seed)
    Y = sample_df[y].to_numpy()
    X = sample_df[x].to_numpy()
    group_ind = sample_df[["x", "group"]].to_numpy()

    with pytest.raises(RuntimeError, match="group_ind must be a 1d array"):
        cd_func.condind(
            X=X,
            Y=Y,
            group_ind=group_ind,
            random_seed=seed,
        )

    # all the group indicators have different values now from 0/1
    sample_df["group"] = sample_df["group"] + 3
    group_ind = sample_df[["group"]].to_numpy()
    group_ind = rng.integers(0, 10, size=len(group_ind))

    with pytest.raises(RuntimeError, match="There should only be two groups"):
        cd_func.condind(
            X=X,
            Y=Y,
            group_ind=group_ind,
            random_seed=seed,
        )

    group_ind = sample_df[["group"]].to_numpy().squeeze()

    # test pre-fit propensity scores, or custom propensity model
    with pytest.raises(
        ValueError, match="Both propensity model and propensity estimates are specified"
    ):
        cd_func.condind(
            X=X,
            Y=Y,
            group_ind=group_ind,
            propensity_model=RandomForestClassifier(),
            propensity_weights=[0.5, 0.5],
            random_seed=seed,
        )

    Y = sample_df[y].to_numpy()
    X = sample_df[x].to_numpy()
    with pytest.raises(ValueError, match="There are 3 group pre-defined estimates"):
        cd_func.condind(
            X=X,
            Y=Y,
            group_ind=group_ind,
            propensity_weights=np.ones((10, 3)) * 0.5,
            random_seed=seed,
        )

    with pytest.raises(ValueError, match="There are 100 pre-defined estimates"):
        cd_func.condind(
            X=X,
            Y=Y,
            group_ind=group_ind,
            propensity_weights=np.ones((100, 2)) * 0.5,
            random_seed=seed,
        )


@pytest.mark.parametrize(
    ["cd_func", "cd_kwargs"],
    [
        [bregman, dict()],
        [kcd, dict()],
        [bregman, {"propensity_model": RandomForestClassifier(n_estimators=50, random_state=seed)}],
        [bregman, {"propensity_weights": np.ones((n_samples, 2)) * 0.5}],
        [kcd, {"propensity_model": RandomForestClassifier(n_estimators=50, random_state=seed)}],
        [kcd, {"propensity_weights": np.ones((n_samples, 2)) * 0.5}],
    ],
)
def test_given_single_environment_when_conditional_ksample_fail_to_reject_null(cd_func, cd_kwargs):
    """Test conditional discrepancy tests on a single environment.

    Here the distributions of each variable are the same across groups of data.
    """
    df = _single_env_scm(n_samples=n_samples, offset=2.0, seed=rng)

    group_col = "group"
    alpha = 0.1
    null_sample_size = 25

    res = cd_func.condind(
        X=df["x"],
        Y=df["x1"],
        group_ind=df[group_col],
        null_sample_size=null_sample_size,
        **cd_kwargs,
        random_seed=seed,
    )
    assert res.pvalue > alpha, f"Fails with {res.pvalue} not greater than {alpha}"
    res = cd_func.condind(
        X=df["x"],
        Y=df["z"],
        group_ind=df[group_col],
        null_sample_size=null_sample_size,
        n_jobs=-1,
        **cd_kwargs,
        random_seed=seed,
    )
    assert res.pvalue > alpha, f"Fails with {res.pvalue} not greater than {alpha}"
    res = cd_func.condind(
        X=df["x"],
        Y=df["y"],
        group_ind=df[group_col],
        null_sample_size=null_sample_size,
        n_jobs=-1,
        **cd_kwargs,
        random_seed=seed,
    )
    assert res.pvalue > alpha, f"Fails with {res.pvalue} not greater than {alpha}"


@pytest.mark.parametrize(
    ["cd_func", "cd_kwargs"],
    [
        [bregman, dict()],
        [kcd, dict()],
        [bregman, {"propensity_model": RandomForestClassifier(n_estimators=50, random_state=seed)}],
        [kcd, {"propensity_model": RandomForestClassifier(n_estimators=50, random_state=seed)}],
    ],
)
def test_cd_simulation_multi_environment(cd_func, cd_kwargs):
    """Test conditional discrepancy tests with multiple environments.

    In this setting, we form a selection diagram

    X1 -> Y <- X and Y -> Z, with S-nodes pointing to X1 and X. That means
    the distributions of X1 and X are different across groups. Therefore
    we should reject the null hypothesis, except for P(Z | Y).
    """
    df = _multi_env_scm(n_samples=n_samples // 2, offset=2.0, seed=rng)

    group_col = "group"
    alpha = 0.1
    null_sample_size = 20

    res = cd_func.condind(
        X=df[["x"]],
        Y=df[["z"]],
        group_ind=df[group_col],
        null_sample_size=null_sample_size,
        n_jobs=-1,
        random_seed=seed,
        **cd_kwargs,
    )
    assert res.pvalue < alpha, f"Fails with {res.pvalue} not less than {alpha}"
    res = cd_func.condind(
        X=df["x"],
        Y=df["y"],
        group_ind=df[group_col],
        null_sample_size=null_sample_size,
        n_jobs=-1,
        random_seed=seed,
        **cd_kwargs,
    )
    assert res.pvalue < alpha, f"Fails with {res.pvalue} not less than {alpha}"
    res = cd_func.condind(
        X=df["x1"],
        Y=df["z"],
        group_ind=df[group_col],
        null_sample_size=null_sample_size,
        n_jobs=-1,
        random_seed=seed,
        **cd_kwargs,
    )
    assert res.pvalue < alpha, f"Fails with {res.pvalue} not less than {alpha}"

    res = cd_func.condind(
        X=df[["x", "x1"]],
        Y=df[["z"]],
        group_ind=df[group_col],
        null_sample_size=null_sample_size,
        n_jobs=-1,
        random_seed=seed,
        **cd_kwargs,
    )
    assert res.pvalue > alpha, f"Fails with {res.pvalue} not less than {alpha}"
