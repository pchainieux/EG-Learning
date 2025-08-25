import torch
from src.models.ei_rnn import EIRNN, EIConfig

def test_rescale_radius_hits_target():
    m = EIRNN(10, 5, EIConfig(hidden_size=16, spectral_radius=1.1))
    target = 1.25
    with torch.no_grad():
        m.rescale_spectral_radius_(target)
    s = torch.linalg.svdvals(m.W_hh).max().item()
    assert abs(s - target) < 5e-3
