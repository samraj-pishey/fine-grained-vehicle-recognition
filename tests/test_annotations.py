import pytest

from vehicle_recognition.data.annotations import (
    NUM_CLASSES,
    build_class_mapping,
    canonical_box_to_matlab,
    load_class_names,
    matlab_box_to_canonical,
    parse_class_name,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("AM General Hummer SUV 2000", ("AM General", "Hummer SUV", 2000)),
        ("Aston Martin V8 Vantage Coupe 2012", ("Aston Martin", "V8 Vantage Coupe", 2012)),
        ("Land Rover LR2 SUV 2012", ("Land Rover", "LR2 SUV", 2012)),
        ("Mercedes-Benz C-Class Sedan 2012", ("Mercedes-Benz", "C-Class Sedan", 2012)),
        ("Ram C/V Cargo Van Minivan 2012", ("Ram", "C/V Cargo Van Minivan", 2012)),
        ("smart fortwo Convertible 2012", ("smart", "fortwo Convertible", 2012)),
        ("Volkswagen Golf Hatchback 1991", ("Volkswagen", "Golf Hatchback", 1991)),
    ],
)
def test_parse_class_name(name, expected):
    assert parse_class_name(name) == expected


def test_parse_rejects_missing_year():
    with pytest.raises(ValueError):
        parse_class_name("Some Car Coupe")


def test_box_roundtrip():
    # MATLAB 1-based inclusive box covering a 10x5 image -> [0,0,10,5]
    assert matlab_box_to_canonical(1, 1, 10, 5) == (0.0, 0.0, 10.0, 5.0)
    assert canonical_box_to_matlab(0.0, 0.0, 10.0, 5.0) == (1, 1, 10, 5)
    x1, y1, x2, y2 = 39, 116, 569, 375
    assert canonical_box_to_matlab(*matlab_box_to_canonical(x1, y1, x2, y2)) == (x1, y1, x2, y2)


@pytest.mark.data
def test_all_official_names_parse(data_paths):
    names = load_class_names(data_paths.devkit / "cars_meta.mat")
    m = build_class_mapping(names)
    assert len(m) == NUM_CLASSES
    for c in m:
        assert c.make and c.model and 1900 < c.year < 2030
        # round trip: reconstruct the official name verbatim
        assert f"{c.make} {c.model} {c.year}" == c.name
    makes = {c.make for c in m}
    assert len(makes) == 49
