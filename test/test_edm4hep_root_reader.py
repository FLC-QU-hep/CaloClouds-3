""" Module to test the data/edm4hep_root_reader module. """
import numpy as np
import numpy.testing as npt
import os
import uproot
from unittest.mock import patch
import awkward as ak

from pointcloud.data import edm4hep_root_reader


def get_example_file():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, "single_event_example.edm4hep.root")


# probably going to need to add a test file for this one
def test_get_events_filenames(tmpdir):
    # one file
    test_file = get_example_file()
    file_names, events = edm4hep_root_reader.get_events_filenames((test_file,))
    assert len(file_names) == 1
    assert len(events) == 1
    assert file_names[0] == test_file
    MCPID_key = "MCParticles/MCParticles.PDG"
    assert MCPID_key in events[0].keys()
    pdgs = events[0][MCPID_key].array()
    assert len(pdgs) == 1
    assert 22 in pdgs[0]
    # two files
    file_names, events = edm4hep_root_reader.get_events_filenames(
        [test_file, test_file]
    )
    assert len(file_names) == 2
    assert len(events) == 2
    for i in range(2):
        assert file_names[i] == test_file
        assert MCPID_key in events[i].keys()
        pdgs = events[i][MCPID_key].array()
        assert 22 in pdgs[0]
    # from opened files
    with uproot.open(test_file) as f:
        file_names, events = edm4hep_root_reader.get_events_filenames([f])
        assert len(file_names) == 1
        assert len(events) == 1
        assert file_names[0] == test_file
        assert MCPID_key in events[0].keys()
        pdgs = events[0][MCPID_key].array()
        assert len(pdgs) == 1
        assert 22 in pdgs[0]


# test on the example file
def test_MultiRootReader_exampleFile():
    file_name = get_example_file()
    reader = edm4hep_root_reader.MultiRootReader([file_name])
    npt.assert_array_equal(reader.file_lengths, [1])
    assert len(reader) == 1
    branch_name = "MCParticles/MCParticles.PDG"
    pdg = reader.get(branch_name, 0)
    assert 22 in pdg
    assert len(pdg) == 5
    assert branch_name in reader.keys()


def fake_events_0():
    pdg_branch_name = "MCParticles/MCParticles.PDG"
    pdg = ak.Array([[22, 11, 11, 22, 22], [22, 211, 11]])
    x_branch_name = "MCParticles/MCParticles.vertex.x"
    x = ak.Array([[0, 1, 2, 3, 4], [-1, -2, -3]])
    parent_begin_name = "MCParticles/MCParticles.parents_begin"
    parent_end_name = "MCParticles/MCParticles.parents_end"
    parent_begin = ak.Array([[0, 0, 1, 3, 6], [0, 0, 1]])
    parents_end = ak.Array([[0, 1, 3, 6, 8], [0, 1, 3]])
    children_begin_name = "MCParticles/MCParticles.daughters_begin"
    children_end_name = "MCParticles/MCParticles.daughters_end"
    children_begin = ak.Array([[0, 3, 6, 7, 8], [0, 2, 3]])
    children_end = ak.Array([[3, 6, 7, 8, 8], [2, 3, 3]])

    # the parent indexs can be longer, or empty
    parent_index_branch_name = "_MCParticles_parents/_MCParticles_parents.index"
    parent_index = ak.Array([[0, 0, 1, 0, 1, 2, 1, 3], [0, 0, 1]])
    child_index_branch_name = "_MCParticles_daughters/_MCParticles_daughters.index"
    child_index = ak.Array([[1, 2, 3, 2, 3, 4, 3, 4], [1, 2, 2]])

    ecal_pdg_name = (
        "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.PDG"
    )
    ecal_pdg = ak.Array([[22], []])
    ecal_x_name = "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.stepPosition.x"
    ecal_x = ak.Array([[0], []])
    ecal_y_name = "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.stepPosition.y"
    ecal_y = ak.Array([[0], []])
    ecal_z_name = "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.stepPosition.z"
    ecal_z = ak.Array([[0], []])
    ecal_energy_name = (
        "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.energy"
    )
    ecal_energy = ak.Array([[1], []])
    ecal_time_name = (
        "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.time"
    )
    ecal_time = ak.Array([[0], []])

    events = {
        pdg_branch_name: pdg,
        x_branch_name: x,
        parent_index_branch_name: parent_index,
        child_index_branch_name: child_index,
        parent_begin_name: parent_begin,
        parent_end_name: parents_end,
        children_begin_name: children_begin,
        children_end_name: children_end,
        ecal_pdg_name: ecal_pdg,
        ecal_x_name: ecal_x,
        ecal_y_name: ecal_y,
        ecal_z_name: ecal_z,
        ecal_energy_name: ecal_energy,
        ecal_time_name: ecal_time,
    }
    return events


def fake_events_1():
    pdg_branch_name = "MCParticles/MCParticles.PDG"
    pdg = ak.Array([[22, 22, 211, 1034, 22], [22], [22, 22, 22, 22]])
    x_branch_name = "MCParticles/MCParticles.vertex.x"
    x = ak.Array([[0.1, 1.1, 2.1, 3.1, 4.1], [-1.1], [-2.1, -3.1, -4.1, -5.1]])
    parent_begin_name = "MCParticles/MCParticles.parents_begin"
    parent_end_name = "MCParticles/MCParticles.parents_end"
    parent_begin = ak.Array([[0, 0, 1, 2, 3], [0], [0, 0, 1, 1]])
    parents_end = ak.Array([[0, 1, 2, 3, 4], [0], [0, 1, 1, 1]])
    children_begin_name = "MCParticles/MCParticles.daughters_begin"
    children_end_name = "MCParticles/MCParticles.daughters_end"
    children_begin = ak.Array([[0, 1, 2, 3, 4], [0], [0, 1, 1, 1]])
    children_end = ak.Array([[1, 2, 3, 4, 4], [0], [1, 1, 1, 1]])
    # the parent indexs can be longer, or empty
    parent_index_branch_name = "_MCParticles_parents/_MCParticles_parents.index"
    parent_index = ak.Array([[0, 1, 2, 3], [], [0]])
    child_index_branch_name = "_MCParticles_daughters/_MCParticles_daughters.index"
    child_index = ak.Array([[1, 2, 3, 4], [], [1]])

    ecal_pdg_name = (
        "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.PDG"
    )
    ecal_pdg = ak.Array([[22, 22], [22], [22, 22]])
    ecal_x_name = "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.stepPosition.x"
    ecal_x = ak.Array([[1, -1], [0], [0, 1]])
    ecal_y_name = "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.stepPosition.y"
    ecal_y = ak.Array([[1, -1], [0], [0, 1]])
    ecal_z_name = "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.stepPosition.z"
    ecal_z = ak.Array([[0, 0], [0], [0, 0]])
    ecal_energy_name = (
        "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.energy"
    )
    ecal_energy = ak.Array([[1, 1], [0], [0, 1]])
    ecal_time_name = (
        "ECalBarrelSiHitsEvenContributions/ECalBarrelSiHitsEvenContributions.time"
    )
    ecal_time = ak.Array([[0, 0], [0], [0, 0]])

    events = {
        pdg_branch_name: pdg,
        x_branch_name: x,
        parent_index_branch_name: parent_index,
        child_index_branch_name: child_index,
        parent_begin_name: parent_begin,
        parent_end_name: parents_end,
        children_begin_name: children_begin,
        children_end_name: children_end,
        ecal_pdg_name: ecal_pdg,
        ecal_x_name: ecal_x,
        ecal_y_name: ecal_y,
        ecal_z_name: ecal_z,
        ecal_energy_name: ecal_energy,
        ecal_time_name: ecal_time,
    }
    return events


def mock_get_events_filenames(file_names):
    if not isinstance(file_names[0], str):
        file_names = [f"mock_file_{i}.edm4hep.root" for i in range(len(file_names))]
    faked = [fake_events_0(), fake_events_1()]
    events = [faked[i % 2] for i in range(len(file_names))]
    return file_names, events


# now don't read anything, just mock the get_events_filenames
# so that the methods can be tested without needing to read the files
@patch(
    "pointcloud.data.edm4hep_root_reader.get_events_filenames",
    new=mock_get_events_filenames,
)
@patch("awkward.Array.array", new=lambda self: self, create=True)
def test_MultiRootReader_mockFiles(mocker):
    reader = edm4hep_root_reader.MultiRootReader(["file1", "file2"])
    assert len(reader) == 5
    pdg = reader.get("MCParticles/MCParticles.PDG", 0)
    npt.assert_array_equal(pdg, [22, 11, 11, 22, 22])
    pdg = reader.get("MCParticles/MCParticles.PDG", 3)
    npt.assert_array_equal(pdg, [22])

    assert reader.decumulated_indices(0) == (0, 0)
    assert reader.decumulated_indices(1) == (0, 1)
    assert reader.decumulated_indices(2) == (1, 0)
    assert reader.decumulated_indices(3) == (1, 1)
    assert reader.decumulated_indices(4) == (1, 2)


def test_Points_exampleFile():
    file_name = get_example_file()
    points = edm4hep_root_reader.Points([file_name])
    expected_parts = [
        "MCParticles",
        "BeamCalCollectionContributions",
        "ECalBarrelSiHitsEvenContributions",
        "ECalBarrelSiHitsOddContributions",
        "EcalEndcapRingCollectionContributions",
        "ECalEndcapSiHitsEvenContributions",
        "ECalEndcapSiHitsOddContributions",
        "HcalBarrelRegCollectionContributions",
        "HCalBarrelRPCHitsContributions",
        "HCalECRingRPCHitsContributions",
        "HcalEndcapRingCollectionContributions",
        "HCalEndcapRPCHitsContributions",
        "HcalEndcapsCollectionContributions",
        "LHCalCollectionContributions",
        "LumiCalCollectionContributions",
        "YokeBarrelCollectionContributions",
        "YokeEndcapsCollectionContributions",
    ]
    assert set(expected_parts) == set(points.parts)
    assert len(points.part_short_names) == len(expected_parts)
    assert len(set(points.part_short_names.values())) < len(expected_parts)
    assert not np.any(["Odd" in part for part in points.part_short_names.values()])
    assert not np.any(["Even" in part for part in points.part_short_names.values()])
    # test getitem
    item0 = points[0]
    length_0 = len(item0["Name"])
    for key in points.keys:
        assert key in item0.keys()
        data = item0[key]
        assert len(data) == length_0
    assert np.all([isinstance(name, str) for name in item0["Name"]])
    assert np.all([part in points.part_short_names.values() for part in item0["Part"]])
    assert np.all([isinstance(pdg, int) for pdg in item0["PDG"]])
    assert np.all([e >= 0 for e in item0["Energy"]])


class MockReader(edm4hep_root_reader.MultiRootReader):
    # only need to override init and get
    # other methods can be inherited
    def __init__(self, file_names):
        self.file_names, self.events = mock_get_events_filenames(file_names)
        self.file_lengths = [len(e["MCParticles/MCParticles.PDG"]) for e in self.events]
        self._cumulative_lengths = np.cumsum(self.file_lengths)

    def get(self, branch_name, index):
        file_idx, idx_in_file = self.decumulated_indices(index)
        return self.events[file_idx][branch_name][idx_in_file]


@patch(
    "pointcloud.data.edm4hep_root_reader.get_events_filenames",
    new=mock_get_events_filenames,
)
@patch("awkward.Array.array", new=lambda self: self, create=True)
def test_Points_mockReader(mocker):
    points = edm4hep_root_reader.Points(["file1", "file2"])
    assert len(points) == 5

    expected_centers = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 1, 0)]
    for i, expected in enumerate(expected_centers):
        found = points.get_ecal_center(i)
        npt.assert_array_equal(found, expected, err_msg=f"center {i}")


def test_Relationships_exampleFile():
    file_name = get_example_file()
    relationships = edm4hep_root_reader.Relationships([file_name])
    assert len(relationships) == 1
    mc_points = relationships.MCParticles(0)
    n_mc_points = len(mc_points["PDG"])
    assert n_mc_points == 5
    for key in ["PDG", "Time", "X", "Y", "Z", "Energy"]:
        assert key in mc_points.keys()
        assert len(mc_points[key]) == n_mc_points

    parent_idxs = relationships.parent_MCParticle_idxs(0)
    assert len(parent_idxs) == 5
    assert len(parent_idxs[0]) == 0
    assert np.all(len(idx) == 1 for idx in parent_idxs[1:])

    child_idxs = relationships.child_MCParticle_idxs(0)
    assert len(child_idxs) == 5
    assert np.all(len(idx) == 1 for idx in child_idxs[:-1])
    assert len(child_idxs[-1]) == 0


@patch(
    "pointcloud.data.edm4hep_root_reader.get_events_filenames",
    new=mock_get_events_filenames,
)
@patch("awkward.Array.array", new=lambda self: self, create=True)
def test_Relationships_mockReader(mocker):
    relationships = edm4hep_root_reader.Relationships(["file1", "file2"])
    assert len(relationships) == 5
    parent_idxs = relationships.parent_MCParticle_idxs(0)
    assert len(parent_idxs) == 5
    npt.assert_array_equal(parent_idxs[0], [])
    npt.assert_array_equal(parent_idxs[1], [0])
    npt.assert_array_equal(parent_idxs[2], [0, 1])
    npt.assert_array_equal(parent_idxs[3], [0, 1, 2])
    npt.assert_array_equal(parent_idxs[4], [1, 3])

    parent_idxs = relationships.parent_MCParticle_idxs(2)
    assert len(parent_idxs) == 5
    npt.assert_array_equal(parent_idxs[0], [])
    for i in range(4):
        npt.assert_array_equal(parent_idxs[i + 1], [i])

    parent_idxs = relationships.parent_MCParticle_idxs(3)
    assert len(parent_idxs) == 1
    npt.assert_array_equal(parent_idxs[0], [])

    child_idxs = relationships.child_MCParticle_idxs(0)
    assert len(child_idxs) == 5
    npt.assert_array_equal(child_idxs[0], [1, 2, 3])
    npt.assert_array_equal(child_idxs[1], [2, 3, 4])
    npt.assert_array_equal(child_idxs[2], [3])
    npt.assert_array_equal(child_idxs[3], [4])

    child_idxs = relationships.child_MCParticle_idxs(2)
    assert len(child_idxs) == 5
    for i in range(4):
        npt.assert_array_equal(child_idxs[i], [i + 1])
    npt.assert_array_equal(child_idxs[4], [])

    child_idxs = relationships.child_MCParticle_idxs(3)
    assert len(child_idxs) == 1
    npt.assert_array_equal(child_idxs[0], [])
