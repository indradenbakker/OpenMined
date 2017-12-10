import numpy as np
import pytest
from syft.syft import FloatTensor, SyftController


def pytest_namespace():
    return {
        'sc': None
    }


@pytest.yield_fixture(autouse=True)
def setup_controller():
    pytest.sc = SyftController()


def test_abs():
    data = np.array([-1., -2., 3., 4., 5., -6.])
    expected = np.array([1., 2., 3., 4., 5., 6.])
    a = pytest.sc.FloatTensor(data)
    b = a.abs()

    np.testing.assert_array_equal(expected, b.to_numpy())
    # a doesn't change
    np.testing.assert_array_equal(data, a.to_numpy())


def test_abs_():
    data = np.array([-1., -2., 3., 4., 5., -6.])
    expected = np.array([1., 2., 3., 4., 5., 6.])
    a = pytest.sc.FloatTensor(data)
    a.abs_()

    # a does change when inlined
    np.testing.assert_array_equal(expected, a.to_numpy())


def test_exp():
    data = np.array([0., 1., 2., 5.])
    expected = np.array([1., 2.71828183, 7.3890561, 148.4131591])
    a = pytest.sc.FloatTensor(data)
    b = a.exp()

    np.testing.assert_almost_equal(expected, b.to_numpy(), decimal=4)
    # a doesn't change
    np.testing.assert_array_equal(data, a.to_numpy())


def test_exp_():
    data = np.array([0., 1., 2., 5.])
    expected = np.array([1., 2.71828183, 7.3890561, 148.4131591])
    a = pytest.sc.FloatTensor(data)
    a.exp_()

    # a does change when inlined
    np.testing.assert_almost_equal(expected, b.to_numpy(), decimal=4)


def test_trace():
    data = np.random.randn(17, 17).astype('float')
    expected = data.trace()

    a = pytest.sc.FloatTensor(data)
    actual = a.trace()
    np.testing.assert_almost_equal(actual, expected, decimal=7)

    a = a.gpu()
    actual = a.trace()
    np.testing.assert_almost_equal(actual, expected, decimal=5)

def test_round():
    data = np.array([12.7292, -3.11, 9.00, 20.4999, 20.5001])
    expected = np.array(13, -3, 9, 20, 21)
    a = pytest.sc.FloatTensor(data)
    b = a.round()

    np.testing.assert_array_equal(expected, b.to_numpy())
    # a doesn't change
    np.testing.assert_array_equal(data, a.to_numpy())

def test_log1p():
    data = np.array([1.2, -0.9, 9.9, 0.1, -0.455])
    expected = np.array([0.78845736, -2.30258509,  2.38876279,  0.09531018, -0.60696948])
    a = pytest.sc.FloatTensor(data)
    b = a.log1p()

    np.testing.assert_almost_equal(expected, b.to_numpy(), decimal=5)