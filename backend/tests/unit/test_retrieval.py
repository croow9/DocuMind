from uuid import uuid4

from app.rag.retrieval import reciprocal_rank_fusion


def test_item_found_by_both_searches_beats_leader_of_one() -> None:
    a, b, c, d = (uuid4() for _ in range(4))

    # c — второй в одном списке и первый в другом; a лидирует только в первом
    merged = reciprocal_rank_fusion([[a, c, b], [c, d]])

    assert merged == [c, a, d, b]


def test_second_place_in_both_lists_beats_first_place_in_one() -> None:
    a, b, c = (uuid4() for _ in range(3))

    merged = reciprocal_rank_fusion([[a, b], [c, b]])

    # b: 1/62 + 1/62 > 1/61 у a и c; a и c равны, выше тот, кто встретился раньше
    assert merged == [b, a, c]


def test_empty_rankings_give_empty_result() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_one_empty_list_keeps_order_of_the_other() -> None:
    a, b, c = (uuid4() for _ in range(3))

    assert reciprocal_rank_fusion([[a, b, c], []]) == [a, b, c]


def test_places_are_counted_from_one() -> None:
    a, b, c = (uuid4() for _ in range(3))

    # при k=0 первое место весит 1/1; если считать места с нуля, будет деление на ноль
    merged = reciprocal_rank_fusion([[a, b], [c, b]], k=0)

    # a: 1, c: 1, b: 1/2 + 1/2 = 1 — ничья, выше тот, кто встретился раньше
    assert merged == [a, b, c]


def test_k_controls_how_much_the_leader_matters() -> None:
    a, c, x, y, z, w, v = (uuid4() for _ in range(7))
    # a — лидер первого списка; c — только четвёртый, но в обоих списках
    rankings = [[a, x, y, c], [z, w, v, c]]

    # малый k: первые места весят много, лидер побеждает
    assert reciprocal_rank_fusion(rankings, k=1)[0] == a
    # большой k сглаживает места, и побеждает согласие списков
    assert reciprocal_rank_fusion(rankings)[0] == c
