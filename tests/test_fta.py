import torch
from fta import FTA


def test_shape_and_sparsity():
    f = FTA(-1, 1, 8, eta=0)
    out = f(torch.tensor([[0.1, -0.9]]))
    assert out.shape == (1, 16)
    assert out[0, :8].sum() == 1 and out[0, 8:].sum() == 1   # one tile per unit when eta = 0


def test_matches_paper_example():
    # Paper: u = 10, k = 4 -> tiles [-10,-5), [-5,0), [0,5), [5,10]; z in (0, 1] always lands in tile 3.
    f = FTA(-10, 10, 4, eta=0)
    for z in (0.01, 0.5, 1.0):
        assert f(torch.tensor([z])).tolist() == [0.0, 0.0, 1.0, 0.0]


def test_fuzzy_gradient():
    f = FTA(-1, 1, 4)                     # eta = delta = 0.5
    z = torch.tensor([0.2], requires_grad=True)
    out = f(z)                            # z sits in tile [0, 0.5]; the next tile [0.5, 1] is 0.3 away
    assert abs(out[3].item() - 0.7) < 1e-6
    out[3].backward()
    assert z.grad.item() == 1.0           # a non-zero gradient, which the hard tiling activation lacks


def test_wide_bound_uses_few_tiles():
    # Discussion example: inputs in [-1, 1] with [l, u] = [-20, 20], delta = 2 hit only 2 of 20 tiles.
    f = FTA(-20, 20, 20, eta=0)
    used = (f(torch.linspace(-1, 0.999, 200).unsqueeze(1)) > 0).any(0).sum()
    assert used == 2
