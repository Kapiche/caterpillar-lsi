# caterpillar: Tests for the caterpillar.analytics.sentiment module
import csv
import os
import shutil
import tempfile

from caterpillar.processing.analysis import stopwords
from caterpillar.processing.analysis.analyse import DefaultAnalyser
from caterpillar.processing.index import IndexWriter, IndexConfig, IndexReader
from caterpillar.processing.schema import ID, TEXT, Schema
from caterpillar.searching.query.querystring import QueryStringQuery
from caterpillar.storage.sqlite import SqliteStorage
from caterpillar_lsi import LSIModel, LSIPlugin
import numpy
from numpy.testing import assert_almost_equal
import pytest


@pytest.fixture
def index_dir(request):
    path = tempfile.mkdtemp()

    def clean():
        shutil.rmtree(path)

    request.addfinalizer(clean)
    return os.path.join(path, "test_index")


def test_lsi_similarity():
    C = numpy.array([[1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1]])
    model = LSIModel(C, 2, False, False)
    for col in C.T:
        # Max similarity should be exact match
        assert_almost_equal(max(model.compare_document(col).values()), 1)

    # Test compare with filter to match the first doc with itself
    assert_almost_equal(model.compare_document(C.T[0], filter_docs=[0])[0], 1)

    with pytest.raises(LSIModel.BadDocumentException):
        # Document's term frequency vector doesn't match model length
        model.compare_document([1])
    with pytest.raises(LSIModel.BadDocumentException):
        # Document's term frequency vector doesn't match model length
        model.classify_document([1])

    with pytest.raises(LSIModel.EmptyDocumentException):
        # Document's term frequency vector is empty
        model.compare_document([0, 0, 0, 0])
    with pytest.raises(LSIModel.EmptyDocumentException):
        # Document's term frequency vector is empty
        model.classify_document([0, 0, 0, 0])


def test_lsi_plugin(index_dir):
    with open(os.path.abspath('caterpillar_lsi/test_resources/csg_data.csv'), 'r') as f:
        analyser = DefaultAnalyser(stopword_list=stopwords.ENGLISH_TEST)
        with IndexWriter(index_dir, IndexConfig(SqliteStorage, Schema(
                text=TEXT(analyser=analyser),
                id=ID(stored=True, indexed=True)
        ))) as writer:
            csv_reader = csv.reader(f)
            i = 0
            doc_ids = []
            for row in csv_reader:
                if row[0] == 'AGEE':  # The Age
                    writer.add_document(frame_size=0, update_index=False, text=row[4], id="doc-{}".format(i))
                    doc_ids.append(i)
                i += 1

    with IndexReader(index_dir) as reader:
        with pytest.raises(RuntimeError):
            # Plugin not yet run
            LSIPlugin(reader).compare_index_with_model(reader, QueryStringQuery("id=doc-{}".format(doc_ids[0])))
    with IndexReader(index_dir) as reader:
        with pytest.raises(RuntimeError):
            # Plugin not yet run
            LSIPlugin(reader).compare_index_using_model(reader)

    with IndexWriter(index_dir) as writer:
        writer.run_plugin(LSIPlugin, normalise_frequencies=True, calculate_document_similarities=False)

    with IndexReader(index_dir) as reader:
        lsi_plugin = LSIPlugin(reader)
        with pytest.raises(RuntimeError):
            # Document similarities not available
            lsi_plugin.get_document_similarities()

    with IndexWriter(index_dir) as writer:
        lsi_plugin = writer.run_plugin(LSIPlugin, normalise_frequencies=True, calculate_document_similarities=True)

    with IndexReader(index_dir) as reader:
        info = LSIPlugin(reader).get_info()

        # Verify that we can produce the same results on the original input matrix
        for d_i in range(info.model.num_documents):
            nv = info.model._normalise_term_vector(lsi_plugin._C.T[d_i])
            fv = info.model._compute_document_feature_vector(nv)
            for fv_i in range(len(fv)):
                assert numpy.isclose(fv[fv_i], info.model.S_Vt_T[d_i][fv_i])

        # Verify documents match each other exactly
        r = lsi_plugin.compare_index_with_model(reader)
        for d_id in r.keys():
            assert_almost_equal(r[d_id][d_id], 1)
        similarities, sim_frame_ids = lsi_plugin.get_document_similarities()
        for i, d_similarities in enumerate(similarities):
            assert_almost_equal(d_similarities[i], 1)

        # Check compare document with model filter
        q = QueryStringQuery('environment')
        r = LSIPlugin(reader).compare_index_with_model(reader, q)
        count = reader.searcher().count(q)
        for d_id in r.keys():
            assert len(r[d_id]) == count

        # Check document similarities with modified order
        f_ids = list(lsi_plugin._frame_ids)
        f_ids.reverse()
        assert lsi_plugin.get_document_similarities(f_ids)[1] == f_ids
        num_features = reader.get_vocab_size() + 1

        # Classify text with the model
        sims = LSIPlugin(reader).compare_index_using_model(reader)
        for f_id in sims.keys():
            assert_almost_equal(sims[f_id][f_id], 1)  # Documents should have a similarity of 1 with themselves

    with IndexWriter(index_dir) as writer:
        with pytest.raises(RuntimeError):
            # Too many features
            writer.run_plugin(LSIPlugin, num_features=num_features)
