import torch
from src.models.ei_rnn import EIRNN, EIConfig
from src.optim.sgd_eg import SGD_EG

def test_dale_columns_sign():
    m = EIRNN(10, 5, EIConfig(hidden_size=32, exc_frac=0.75))
    W = m.W_hh.detach()
    s = m.sign_vec 
    col_signed = W * s
    assert torch.all(col_signed >= -1e-8)

def test_spectral_rescale_preserves_signs():
    m = EIRNN(10, 5, EIConfig(hidden_size=24, exc_frac=0.8))
    s_before = torch.linalg.svdvals(m.W_hh).max()
    m.rescale_spectral_radius_(1.3)
    s_after  = torch.linalg.svdvals(m.W_hh).max()
    assert torch.all((m.W_hh * m.sign_vec) >= -1e-8)
    assert abs(s_after.item() - 1.3) < 5e-3

def test_eg_min_magnitude_floor():
    torch.manual_seed(0)
    H = 20
    m = EIRNN(6, 3, EIConfig(hidden_size=H))
    with torch.no_grad():
        m.W_hh.mul_(1e-8)
    opt = SGD_EG([{"params":[m.W_hh], "update_alg":"eg", "lr":1.0, "min_magnitude":1e-6}])
    m.W_hh.grad = torch.ones_like(m.W_hh)
    opt.step()
    assert (m.W_hh.abs() >= 1e-6 - 1e-12).all()
