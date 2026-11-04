import ipaddress, pytest
from k1_chat import fleet


def test_active_robots_have_valid_ips_in_subnet():
    f = fleet.load(); net = ipaddress.ip_network(f["network"]["subnet"])
    for name, ip in fleet.active(f).items():
        assert ipaddress.ip_address(ip) in net, name


def test_ips_unique():
    ips = list(fleet.active().values()); assert len(ips) == len(set(ips))


def test_lookup_case_insensitive_and_a1_has_no_ip():
    assert fleet.robot_ip("a2") == "192.168.50.215"
    with pytest.raises(KeyError):
        fleet.robot_ip("A1")


def test_expected_robots_present():
    assert set(fleet.load()["robots"]) == {"A1", "A2", "A3", "B1", "B2", "B3"}
