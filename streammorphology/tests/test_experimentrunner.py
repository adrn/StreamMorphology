# coding: utf-8

""" Test the classes in experimentrunner  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import shutil

# Third-party
import numpy as np
import pytest

# Project
from ..config import ConfigNamespace, save
from ..experimentrunner import OrbitGridExperiment, ExperimentRunner

class TestOrbitGridExperiment(object):

    def test_subclassing(self):

        # missing properties and methods - this should fail
        class StupidExperiment(OrbitGridExperiment):
            pass

        with pytest.raises(TypeError) as exc:
            exp = StupidExperiment()
            assert "error_codes" in exc
            assert "dache_dtype" in exc
            assert "_run_kwargs" in exc
            assert " run" in exc # space there to distinguish from _run_kwargs

        # ----------------------------------------------
        # try by passing kwargs
        # this should succeed
        test_path = '/tmp/stupid-experiment'
        test_defaults = dict(
            test=0.,
            cache_filename='test.npy',
            w0_filename='w0.npy',
            potential_filename='potential.yml'
        )

        # create a dummy initial conditions file...
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        np.save(os.path.join(test_path, test_defaults['w0_filename']),
                np.array([[1.,2,3,4,5,6]]))

        class StupidExperiment(OrbitGridExperiment):
            _run_kwargs = []
            error_codes = dict()
            cache_dtype = [('test', 'f8')]
            config_defaults = test_defaults

            @classmethod
            def run(cls, w0, potential):
                pass

        # nothing set
        with StupidExperiment(test_path, **test_defaults) as exp:
            tmpdir = exp._tmpdir
            assert os.path.exists(tmpdir)
            assert exp.config.test == 0.

        # --

        kwargy = dict(
            cache_filename='test.npy',
            w0_filename='w0.npy',
            potential_filename='potential.yml'
        )

        # nothing set
        with StupidExperiment(test_path, **kwargy) as exp:
            tmpdir = exp._tmpdir
            assert os.path.exists(tmpdir)
            assert exp.config.test == 0.

        assert not os.path.exists(tmpdir)
        shutil.rmtree(test_path)

        # ----------------------------------------------
        # this should succeed
        test_path = '/tmp/stupid-experiment'
        test_defaults = dict(
            test=0.,
            cache_filename='test.npy',
            w0_filename='w0.npy',
            potential_filename='potential.yml'
        )

        # create a dummy initial conditions file...
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        np.save(os.path.join(test_path, test_defaults['w0_filename']),
                np.array([[1.,2,3,4,5,6]]))

        # create a dumby config file
        ns = ConfigNamespace()
        for k,v in test_defaults.items():
            setattr(ns, k, v)
        save(ns, os.path.join(test_path, "test.cfg"))

        class StupidExperiment2(OrbitGridExperiment):
            _run_kwargs = []
            error_codes = dict()
            cache_dtype = [('test', 'f8')]
            config_defaults = test_defaults

            @classmethod
            def run(cls, w0, potential):
                pass

        with StupidExperiment2.from_config(test_path,
                                           config_filename='test.cfg') as exp:
            tmpdir = exp._tmpdir
            assert os.path.exists(tmpdir)
            assert exp.config.test == 0.

        assert not os.path.exists(tmpdir)
        shutil.rmtree(test_path)

class TestExperimentRunner(object):
    # TODO: no tests right now cause I *suck*!
    pass

