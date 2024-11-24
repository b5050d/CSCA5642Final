from scratch import (
    normalize_cart_position,
    normalize_cart_speed,
    normalize_pole_angle,
    normalize_pole_angular_velocity,
    normalize_data,
    )
import numpy as np
def test_normalize_cart_position():
    a = -4.8
    ans = normalize_cart_position(a)
    assert ans == -1

    a = 4.8
    ans = normalize_cart_position(a)
    assert ans == 1

    a = 0
    ans = normalize_cart_position(a)
    assert ans == 0

    a = 2.4
    ans = normalize_cart_position(a)
    assert ans == .5

def test_normalize_cart_speed():
    a = -5
    ans = normalize_cart_speed(a)
    assert ans == -1

    a = 5
    ans = normalize_cart_speed(a)
    assert ans == 1

    a = 0
    ans = normalize_cart_speed(a)
    assert ans == 0

    a = 2.5
    ans = normalize_cart_speed(a)
    assert ans == .5

def test_normalize_pole_angle():
    a = -.418
    ans = normalize_pole_angle(a)
    assert ans == -1

    a = .418
    ans = normalize_pole_angle(a)
    assert ans == 1

    a = 0
    ans = normalize_pole_angle(a)
    assert ans == 0

    a = .209
    ans = normalize_pole_angle(a)
    assert ans == .5

def test_normalize_pole_angular_velocity():
    a = -5
    ans = normalize_pole_angular_velocity(a)
    assert ans == -1

    a = 5
    ans = normalize_pole_angular_velocity(a)
    assert ans == 1

    a = 0
    ans = normalize_pole_angular_velocity(a)
    assert ans == 0

    a = 2.5
    ans = normalize_pole_angular_velocity(a)
    assert ans == .5

def test_normalize_data():
    a = np.array([[-4.8,-5,-.418,-5]])
    ans = normalize_data(a)
    assert ans[0][0] == -1
    assert ans[0][1] == -1
    assert ans[0][2] == -1
    assert ans[0][3] == -1

    a = narray([[4.8,5,.418,5]])
    ans = normalize_data(a)
    assert ans[0][0] == 1
    assert ans[0][1] == 1
    assert ans[0][2] == 1
    assert ans[0][3] == 1


    a = np.array([[0,0,0,0]])
    ans = normalize_data(a)
    assert ans[0][0] == 0
    assert ans[0][1] == 0
    assert ans[0][2] == 0
    assert ans[0][3] == 0

    a = np.array([[2.4,2.5,.209,2.5]])
    ans = normalize_data(a)
    assert ans[0][0] == .5
    assert ans[0][1] == .5
    assert ans[0][2] == .5
    assert ans[0][3] == .5