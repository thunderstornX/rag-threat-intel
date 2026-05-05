from ingest.document import Document, DocSource


def test_fingerprint_is_stable_for_identical_inputs():
    a = Document(text="hi", source=DocSource.NVD, source_id="X")
    b = Document(text="hi", source=DocSource.NVD, source_id="X")
    assert a.fingerprint == b.fingerprint


def test_fingerprint_changes_when_text_changes():
    a = Document(text="hi", source=DocSource.NVD, source_id="X")
    b = Document(text="hi!", source=DocSource.NVD, source_id="X")
    assert a.fingerprint != b.fingerprint


def test_to_dict_roundtrips_essential_fields():
    a = Document(text="hi", source=DocSource.PDF, source_id="x.pdf:p1",
                  metadata={"page": 1})
    d = a.to_dict()
    assert d["source"] == "pdf"
    assert d["source_id"] == "x.pdf:p1"
    assert d["metadata"] == {"page": 1}
