from agent.run import _status, cited_labels

CHUNKS = [{"citation": "[equipment_policy.md#Personal laptops]"}]


def test_cited_labels_split_known_and_invented():
    answer = ("Not allowed [equipment_policy.md#Personal laptops]. "
              "Also [made_up.md#Nowhere].")
    assert cited_labels(answer, CHUNKS) == (["[equipment_policy.md#Personal laptops]"],
                                            ["[made_up.md#Nowhere]"])


def test_status_comes_from_tools_and_citations():
    valid = ["[equipment_policy.md#Personal laptops]"]
    assert _status(CHUNKS, [], valid, []) == "completed"
    assert _status(CHUNKS, [], [], []) == "needs_human"
    assert _status(CHUNKS, [], valid, ["[x.md#y]"]) == "needs_human"
    assert _status([], ["search failed"], [], []) == "failed"
