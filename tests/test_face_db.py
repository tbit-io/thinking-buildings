import numpy as np
import pytest

from thinking_buildings.face_db import FaceDB


@pytest.fixture
def db(tmp_path):
    db = FaceDB(str(tmp_path / "test.db"))
    yield db
    db.close()


class TestAddPerson:
    def test_creates_person_returns_id(self, db):
        pid = db.add_person("alice")
        assert isinstance(pid, int)
        assert pid >= 1

    def test_idempotent_on_duplicate(self, db):
        pid1 = db.add_person("alice")
        pid2 = db.add_person("alice")
        assert pid1 == pid2

    def test_different_names_get_different_ids(self, db):
        pid1 = db.add_person("alice")
        pid2 = db.add_person("bob")
        assert pid1 != pid2


class TestAddEmbedding:
    def test_stores_and_caches(self, db):
        emb = np.random.randn(512).astype(np.float32)
        db.add_embedding("alice", emb)
        cached = db.get_all_embeddings()
        assert "alice" in cached
        assert len(cached["alice"]) == 1
        np.testing.assert_array_almost_equal(cached["alice"][0], emb)

    def test_auto_creates_person(self, db):
        emb = np.random.randn(512).astype(np.float32)
        db.add_embedding("bob", emb)
        persons = db.list_persons()
        names = [name for name, _ in persons]
        assert "bob" in names

    def test_multiple_embeddings_per_person(self, db):
        for _ in range(3):
            db.add_embedding("alice", np.random.randn(512).astype(np.float32))
        assert len(db.get_all_embeddings()["alice"]) == 3


class TestGetAllEmbeddings:
    def test_returns_cached_dict(self, db):
        db.add_embedding("alice", np.random.randn(512).astype(np.float32))
        db.add_embedding("bob", np.random.randn(512).astype(np.float32))
        result = db.get_all_embeddings()
        assert set(result.keys()) == {"alice", "bob"}

    def test_correct_shapes(self, db):
        emb = np.random.randn(512).astype(np.float32)
        db.add_embedding("alice", emb)
        cached = db.get_all_embeddings()
        assert cached["alice"][0].shape == (512,)
        assert cached["alice"][0].dtype == np.float32


class TestRemovePerson:
    def test_deletes_person_and_embeddings(self, db):
        db.add_embedding("alice", np.random.randn(512).astype(np.float32))
        result = db.remove_person("alice")
        assert result is True
        assert "alice" not in db.get_all_embeddings()
        assert db.list_persons() == []

    def test_returns_false_for_unknown(self, db):
        assert db.remove_person("nonexistent") is False

    def test_updates_cache(self, db):
        db.add_embedding("alice", np.random.randn(512).astype(np.float32))
        db.add_embedding("bob", np.random.randn(512).astype(np.float32))
        db.remove_person("alice")
        assert "alice" not in db.get_all_embeddings()
        assert "bob" in db.get_all_embeddings()


class TestListPersons:
    def test_returns_name_count_tuples(self, db):
        db.add_embedding("alice", np.random.randn(512).astype(np.float32))
        db.add_embedding("alice", np.random.randn(512).astype(np.float32))
        db.add_embedding("bob", np.random.randn(512).astype(np.float32))
        persons = db.list_persons()
        assert ("alice", 2) in persons
        assert ("bob", 1) in persons

    def test_sorted_by_name(self, db):
        db.add_person("charlie")
        db.add_person("alice")
        db.add_person("bob")
        persons = db.list_persons()
        names = [name for name, _ in persons]
        assert names == ["alice", "bob", "charlie"]
