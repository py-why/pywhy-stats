import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from pywhy_stats import bregman, kcd

seed = 12345
rng = np.random.default_rng(seed)
# number of samples to use in generating test dataset; the lower the faster
n_samples = 150


def single_env_scm(n_samples=200, offset=0.0):
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


def multi_env_scm(n_samples=100, offset=1.5):
    df = single_env_scm(n_samples=n_samples)
    df["group"] = 0

    new_df = single_env_scm(n_samples=n_samples, offset=offset)
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
    sample_df = single_env_scm(n_samples=10)
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
        # [kcd, {"l2": 1e-3}],
        # [kcd, {"l2": (1e-3, 2e-3)}],
    ],
)
@pytest.mark.parametrize(
    ["df", "env_type"],
    [
        [single_env_scm(n_samples=n_samples, offset=2.0), "single"],
        [multi_env_scm(n_samples=n_samples // 2, offset=2.0), "multi"],
    ],
)
def test_cd_simulation(cd_func, df, env_type, cd_kwargs):
    """Test conditional discrepancy tests."""
    group_col = "group"
    alpha = 0.1
    null_sample_size = 25

    if env_type == "single":
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
    elif env_type == "multi":
        res = cd_func.condind(
            X=df[["x"]],
            Y=df[["z"]].copy(),
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
