import input_output as io
import pandas as pd
import pytest


class Test_IO:
    @pytest.mark.parametrize(
        "fname,expected_shape",
        [
            ("data/efield.t",(101, 3)),
            ("data/expec.t",(101, 5)),
            ("data/npop.t",(101, 38)),
            ("data/nstate_i.t",(101, 480)),
            ("data/table.dat",(4950, 6))
        ],
    )
    def test_table_input(self, fname, expected_shape):
        assert io.read_table(fname).shape == expected_shape
