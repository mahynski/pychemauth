"""
Unittests for mypy typing.

author: nam
credit: @bbarker, https://gist.github.com/bbarker/4ddf4a1c58ae8465f3d37b6f2234a421
"""
import os
import glob
import subprocess
import unittest
import pytest

from typing import List


class MyPyTest(unittest.TestCase):
    """Perform type checking with mypy."""

    def __init__(self, *args, **kwargs) -> None:
        """Instantiate the class."""
        self.pkgname: str = "pychemauth"
        super(MyPyTest, self).__init__(*args, **kwargs)
        my_env = os.environ.copy()
        self.pypath: str = my_env.get("PYTHONPATH", os.getcwd())
        self.mypy_opts: List[str] = ['--ignore-missing-imports']

    def test_run_mypy_module(self):
        """Run mypy on all module files."""
        mypy_call: List[str] = ["mypy"] + self.mypy_opts + ["-p", self.pkgname]
        browse_result: int = subprocess.call(mypy_call, env=os.environ, cwd=self.pypath)
        self.assertEqual(browse_result, 0, 'mypy found errors')

    @pytest.mark.skip(
        reason="At the moment it is not a priority to enforce type checking on the tests."
    )
    def test_run_mypy_tests(self):
        """Run mypy on all tests in module under the tests directory"""
        for test_file in glob.iglob(f'{os.getcwd()}/tests/**/*.py', recursive=True):
            mypy_call: List[str] = ["mypy"] + self.mypy_opts + [test_file]
            test_result: int = subprocess.call(mypy_call, env=os.environ, cwd=self.pypath)
            self.assertEqual(test_result, 0, f'mypy on test {test_file}')