from numpy.testing import assert_allclose


def equal(x, y, t):
    assert_allclose(x, y, atol=t)
