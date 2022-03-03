import input_output as io
import pandas as pd
import pytest


class Test_IO:
    @pytest.mark.parametrize(
        "fname",
        [
            "data/efield.t",
            "data/expec.t",
            "data/npop.t",
            "data/nstate_i.t",
            "data/table.dat",
        ],
    )
    def test_table_input(self, fname):
        assert type(io.read_table(fname)) == pd.DataFrame
