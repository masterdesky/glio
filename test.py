# Language: Python
import os
import tempfile
import unittest
import numpy as np

from glio.fortranio import FortranFile, FortranIOException
from glio.snapshot import SnapshotHeader, SnapshotBase, SnapshotIOException


dummy_header_schema = {
    'npart': ('i4', 6),
    'mass': ('f8', 6)
}
dummy_blocks_schema = {
    'dummy': ('f4', 1, [0, 1, 2], True)
}

class DummySnapshot(SnapshotBase):
    def __init__(self, fname, header_schema, blocks_schema, ptype_aliases=None):
        super(DummySnapshot, self).__init__(fname, header_schema, blocks_schema, ptype_aliases)

    # For testing, assume all requested particle types exist.
    def _block_exists(self, name, ptypes):
        return True

    # Dummy _parse_block: simply splits the block_data into equal chunks for each valid particle type.
    def _parse_block(self, block_data, name, dtype, ndims, ptypes):
        pdata = []
        total_items = block_data.size
        num_ptypes = len(self.ptype_indices)
        # For test purposes, simply divide equally among particle types (if possible).
        items_per_ptype = total_items // num_ptypes if num_ptypes else 0
        begin = 0
        for p in self.ptype_indices:
            if p in ptypes:
                end = begin + items_per_ptype
                parray = block_data[begin:end]
                if ndims > 1 and parray.size > 0:
                    parray.shape = (-1, ndims)
                pdata.append(parray)
                begin = end
            else:
                pdata.append(None)
        return pdata

class TestFortranFile(unittest.TestCase):
    def test_write_and_read_record(self):
        # Create a temporary file to write a simple record.
        rec = np.array([1, 2, 3], dtype='i4')
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            fname = tmp.name
        try:
            # Write the record using FortranFile.
            with FortranFile(fname, 'wb') as ffile:
                ffile.write_ndarray(rec)
            # Read it back.
            with FortranFile(fname, 'rb') as ffile:
                read_rec = ffile.read_record('i4')
            np.testing.assert_array_equal(read_rec, rec)
        finally:
            os.remove(fname)
    
    def test_invalid_control_byte(self):
        with self.assertRaises(ValueError):
            FortranFile("dummy", control_bytes='invalid')

class TestSnapshotHeader(unittest.TestCase):
    def test_init_and_verify(self):
        # Create a dummy header.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            fname = tmp.name
        try:
            hdr = SnapshotHeader(fname, dummy_header_schema)
            # The fields should be set to zeros.
            self.assertTrue(hasattr(hdr, 'npart'))
            self.assertTrue(hasattr(hdr, 'mass'))
            self.assertEqual(np.asarray(hdr.npart).sum(), 0)
        finally:
            os.remove(fname)

class TestDummySnapshot(unittest.TestCase):
    def setUp(self):
        # Create a temporary file with a dummy header record.
        self.tmp_file = tempfile.NamedTemporaryFile(delete=False)
        self.fname = self.tmp_file.name
        # Prepare a header: npart and mass.
        # For simplicity, set npart to fixed counts and mass to zero.
        npart = np.array([10, 20, 30, 0, 0, 0], dtype='i4')
        mass = np.array([0, 0, 0, 0, 0, 0], dtype='f8')
        header_bytes = npart.tobytes() + mass.tobytes()
        # Write header using a Fortran record: write record start, data, and record end.
        with open(self.fname, 'wb') as f:
            ctrl = np.array([len(header_bytes)], dtype='i4')
            ctrl.tofile(f)
            f.write(header_bytes)
            ctrl.tofile(f)
        # Create a dummy block record data for the "dummy" block.
        # For simplicity, write a record of 30 float32 numbers.
        self.block_data = np.arange(30, dtype='f4')
        with open(self.fname, 'ab') as f:
            block_bytes = self.block_data.tobytes()
            ctrl = np.array([len(block_bytes)], dtype='i4')
            ctrl.tofile(f)
            f.write(block_bytes)
            ctrl.tofile(f)

    def tearDown(self):
        os.remove(self.fname)

    def test_load_and_parse(self):
        # Create a DummySnapshot instance.
        snap = DummySnapshot(self.fname, dummy_header_schema, dummy_blocks_schema, ptype_aliases={'type0': 0})
        # Load header and dummy block.
        snap.load()
        # Check that header.npart has been loaded correctly.
        np.testing.assert_array_equal(snap.header.npart, np.array([10, 20, 30, 0, 0, 0], dtype='i4'))
        # Test that dummy block is parsed into list with one ndarray per particle type.
        self.assertEqual(len(snap.dummy), len(list(snap.ptype_indices)))
        # For types not in the schema's ptypes ([0, 1, 2]), the entry should be None.
        for p in snap.ptype_indices:
            if p not in dummy_blocks_schema['dummy'][2]:
                self.assertIsNone(snap.dummy[p])
            else:
                # For testing, check that the ndarray has the proper dtype.
                self.assertEqual(snap.dummy[p].dtype, np.dtype('f4'))

if __name__ == '__main__':
    unittest.main()