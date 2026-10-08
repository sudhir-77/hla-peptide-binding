import numpy as np
import torch

from src.models import PanAlleleCNN, SmallTransformer
from src.train import predict, train_regressor


def test_models_return_unit_interval_scores():
    torch.manual_seed(0)
    pep = torch.randint(0, 20, (4, 15))
    mhc = torch.randint(0, 20, (4, 34))
    cnn = PanAlleleCNN(hidden=8, dropout=0.0)
    transformer = SmallTransformer(
        d_model=16, nhead=2, nlayers=1, dim_feedforward=32, dropout=0.0, max_pep_len=15, max_mhc_len=34
    )
    for model in (cnn, transformer):
        out = model(pep, mhc)
        assert out.shape == (4,)
        assert torch.all((out > 0) & (out < 1))


def test_predict_empty_and_short_training_separates_easy_peptides():
    device = torch.device("cpu")
    model = PanAlleleCNN(hidden=8, dropout=0.0)
    empty = predict(model, np.zeros((0, 15), dtype=np.int64), np.zeros((0, 34), dtype=np.int64), device)
    assert empty.shape == (0,)

    n = 40
    pep = np.full((n, 15), 21, dtype=np.int64)  # PAD
    mhc = np.zeros((n, 34), dtype=np.int64)
    y = np.zeros(n, dtype=np.float64)
    binders = np.zeros(n, dtype=np.int64)
    # Even rows are poly-A binders, odd rows are poly-C non-binders, in both splits.
    for i in range(n):
        if i % 2 == 0:
            pep[i, :9] = 0
            y[i] = 0.9
            binders[i] = 1
        else:
            pep[i, :9] = 1
            y[i] = 0.05
    history = train_regressor(
        model,
        pep[:30],
        mhc[:30],
        y[:30],
        binders[:30],
        pep[30:],
        mhc[30:],
        y[30:],
        binders[30:],
        epochs=6,
        patience=6,
        batch_size=16,
        lr=1e-2,
        device=device,
        seed=0,
    )
    assert history.best_val_auroc >= 0.8
