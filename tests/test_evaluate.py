import json

from app.evaluate import evaluate, match_titles, norm_title
from app.schema import ClaimPacket, Room, Sweep
from tests.conftest import make_book, make_item


def test_title_normalisation_and_matching():
    assert norm_title("The Hobbit: or There and Back Again") == "hobbit"
    m = match_titles(["Sapiens", "The Hobbit", "Wrong Book"], ["hobbit", "Sapiens: A Brief History"])
    assert m == {0: 1, 1: 0}


def test_evaluate_end_to_end(tmp_path):
    (tmp_path / "books.csv").write_text(
        "shelf,position,title,author,legible,orientation,height_cm,thickness_cm\n"
        "A1,1,Sapiens,,yes,upright,24,3\n"
        "A1,2,The Hobbit,,yes,upright,20,2\n"
        "A1,3,,,no,upright,,\n"
    )
    (tmp_path / "prices.csv").write_text(
        "title,isbn,replacement_cost,used_value,currency,source,url,checked_at\nSapiens,,500,,INR,x,y,z\n"
    )
    (tmp_path / "items.csv").write_text("name,category,w_cm,h_cm,d_cm,notes\nLamp,lighting,,,,\nRug,rug,,,,\n")
    (tmp_path / "room.json").write_text(json.dumps({"floor_area_m2": 12.0, "wall_area_m2": 40.0}))

    packet = ClaimPacket(
        sweep=Sweep(id="s", captured_at="x"),
        room=Room(floor_area_m2=12.6, wall_area_m2=50.0),
        books=[
            make_book(1, title="Sapiens", repl=550, spine_height_cm=23.0),
            make_book(2, title="The Hobbit", spine_height_cm=21.0),
            make_book(3, status="unidentified"),
        ],
        items=[make_item(1, low=1, high=1)],
    )
    res = {m.measure: m for m in evaluate(packet, tmp_path, {"sweep_ended_t": 100, "packet_ready_t": 250})}
    assert res["Book count"].passed is True
    assert res["Title identification"].passed is True
    assert res["Book prices"].passed is True  # 10% off
    assert res["Non-book items"].passed is False  # 1 of 2
    assert res["Floor area"].passed is True  # 5% off
    assert res["Wall area"].passed is False  # 25% off
    assert res["Time to packet"].passed is True
