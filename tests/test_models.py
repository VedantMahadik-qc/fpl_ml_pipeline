import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from fpl_models import SeedAveragedClassifier, SeedAveragedRegressor, TwoStageModel
from fpl_train import TrainConfig, build_regressor, fit_ensemble


def toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    return X, (X[:, 0] > 0).astype(int), 2 * X[:, 1] + rng.normal(scale=0.1, size=n)


def test_seed_averaged_predictions_are_member_means():
    X, y, r = toy_data()
    clfs = [XGBClassifier(n_estimators=10, subsample=0.8, random_state=s).fit(X, y)
            for s in range(3)]
    regs = [XGBRegressor(n_estimators=10, subsample=0.8, random_state=s).fit(X, r)
            for s in range(3)]
    clf, reg = SeedAveragedClassifier(clfs), SeedAveragedRegressor(regs)

    np.testing.assert_allclose(clf.predict_proba(X),
                               np.mean([c.predict_proba(X) for c in clfs], axis=0))
    np.testing.assert_allclose(reg.predict(X), np.mean([g.predict(X) for g in regs], axis=0))
    model = TwoStageModel(clf, reg, ["a", "b", "c"])
    prob, conditional = model.predict_parts(X)
    np.testing.assert_allclose(model.predict(X), prob * conditional)


def test_ensemble_of_one_is_the_plain_estimator():
    X, _, r = toy_data()
    config = TrainConfig(n_estimators=10, ensemble_size=1)
    single = fit_ensemble(build_regressor, X[:200], r[:200], X[200:], r[200:], config, "reg")
    assert isinstance(single, XGBRegressor)

    config = TrainConfig(n_estimators=10, ensemble_size=3)
    bagged = fit_ensemble(build_regressor, X[:200], r[:200], X[200:], r[200:], config, "reg")
    assert isinstance(bagged, SeedAveragedRegressor) and len(bagged.estimators) == 3
    seeds = {member.random_state for member in bagged.estimators}
    assert seeds == {4200, 4201, 4202}
