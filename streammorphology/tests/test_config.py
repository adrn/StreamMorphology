# coding: utf-8

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Project
from ..config import ConfigNamespace, save, load

def test_create():
    ns = ConfigNamespace()
    ns.derp = 15.
    ns.derp1 = "Cat"
    ns.derp2 = None
    ns.derp3 = ['one', 'two', 'three']

    print(dict(**ns))

def test_attribute_access():
    ns = ConfigNamespace()
    ns.derp = 15.
    ns.derp1 = "Cat"
    ns.derp2 = None
    ns.derp3 = ['one', 'two', 'three']

    assert ns.derp == 15.
    assert ns.derp1 == "Cat"
    assert ns.derp2 is None

def test_save_load():
    filename = "/tmp/test_config.yml"

    ns = ConfigNamespace()
    ns.derp = 15.
    ns.derp1 = "Cat"
    ns.derp2 = None
    ns.derp3 = ['one', 'two', 'three']

    with open(filename, "w") as f:
        save(ns, f)

    save(ns, filename)

    ns = load(filename)
    assert ns.derp == 15.
    assert ns.derp1 == "Cat"
    assert ns.derp2 is None
