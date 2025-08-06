import torch
from src.data import mod_cog_tasks as mct
from neurogym import Dataset
from src.training.losses import decision_mask_from_inputs

def _one_batch(task="dm1", batch_size=8, seq_len=40):
    env = getattr(mct, task)()
    ds = Dataset(env, batch_size=batch_size, seq_len=seq_len, batch_first=True)
    X, Y = ds()
    return torch.from_numpy(X).float(), torch.from_numpy(Y).long(), env

def test_shapes_and_mask():
    X, Y, env = _one_batch("dm1", 8, 60)
    B,T,D = X.shape
    assert Y.shape == (B,T)
    assert D >= 1
    mask = decision_mask_from_inputs(X, 0.5)
    assert mask.any().item() and (~mask).any().item()

def test_action_dim_consistency():
    _, _, env = _one_batch("dm1", 4, 20)
    assert env.action_space.n >= 2 
