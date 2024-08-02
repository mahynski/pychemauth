"""
Unittests for fastnumpyio.

author: nam
"""
import unittest
import io
import tempfile
import os

import numpy as np

from pychemauth.utils import fastnumpyio


class Checkfastnumpyio(unittest.TestCase):
    """Check fastnumpyio gives same results as numpy."""

    def test_io(self):
        """Test read/write in numpy is the same as in fastnumpyio."""
        testarray = np.random.rand(3, 64, 64).astype("float32")

        # Numpy reference when it saves data and reloads it
        buffer = io.BytesIO()
        np.save(buffer, testarray)
        numpy_save_data = buffer.getvalue()

        buffer = io.BytesIO(numpy_save_data)
        test_numpy_save = np.load(buffer)

        # fastnumpyio when it saves the same data and reloads it
        buffer = io.BytesIO()
        fastnumpyio.save(buffer, testarray)
        fastnumpyio_save_data = buffer.getvalue()

        buffer = io.BytesIO(fastnumpyio_save_data)
        test_fastnumpyio_save = fastnumpyio.load(buffer)

        # Packing and unpacking
        fastnumpyio_pack_data = fastnumpyio.pack(testarray)
        test_fastnumpyio_pack = fastnumpyio.unpack(fastnumpyio_pack_data)

        np.array_equal(test_numpy_save, test_fastnumpyio_save)
        np.array_equal(test_numpy_save, test_fastnumpyio_pack)

    def test_disk(self):
        """Test reading and writing to disk."""
        testarray = np.random.rand(3, 64, 64).astype("float32")
        
        dir_ = tempfile.TemporaryDirectory()
        with open(os.path.join(dir_.name, 'dummy.npy'), 'wb') as f:
            fastnumpyio.save(f, testarray)
            
        with open(os.path.join(dir_.name, 'dummy.npy'), 'rb') as f:
            checkarray = fastnumpyio.load(f)
        dir_.cleanup()

        np.testing.assert_allclose(testarray, checkarray)
