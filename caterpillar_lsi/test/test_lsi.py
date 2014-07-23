# caterpillar: Tests for the caterpillar.analytics.sentiment module
#
#
import csv
import os

import numpy
from numpy.testing import assert_almost_equal
import pytest
from caterpillar.data.sqlite import SqliteMemoryStorage
from caterpillar.processing.analysis import stopwords
from caterpillar.processing.analysis.analyse import DefaultAnalyser
from caterpillar.processing.index import Index
from caterpillar.processing.schema import CATEGORICAL_TEXT, ID, TEXT, Schema
from caterpillar.searching.query.querystring import QueryStringQuery
from caterpillar_lsi import LSIModel, LSIPlugin


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

    with pytest.raises(LSIModel.EmptyDocumentException):
        # Document's term frequency vector is empty
        model.compare_document([0, 0, 0, 0])


def test_lsi_plugin():
    storage_cls = SqliteMemoryStorage
    with open(os.path.abspath('caterpillar_lsi/test_resources/csg_data.csv'), 'r') as f:
        analyser = DefaultAnalyser(stopword_list=stopwords.ENGLISH_TEST)
        index = Index.create(Schema(text=TEXT(analyser=analyser), id=ID(stored=True, indexed=True)),
                             storage_cls=storage_cls)
        csv_reader = csv.reader(f)
        i = 0
        doc_ids = []
        for row in csv_reader:
            if row[0] == 'AGEE':  # The Age
                index.add_document(frame_size=0, update_index=False, text=row[4], id="doc-{}".format(i))
                doc_ids.append(i)
            i += 1
        index.reindex()

        with pytest.raises(Exception):
            # Plugin not yet run
            LSIPlugin(index).compare_document(index, QueryStringQuery("id=doc-{}".format(doc_ids[0])))

        lsi_plugin = index.run_plugin(LSIPlugin, normalise_frequencies=True, calculate_document_similarities=False)
        with pytest.raises(Exception):
            # Document similarities not available
            lsi_plugin.get_document_similarities()

        lsi_plugin = index.run_plugin(LSIPlugin, normalise_frequencies=True, calculate_document_similarities=True)
        info = LSIPlugin(index).get_info()

        # Verify that we can produce the same results on the original input matrix
        for d_i in range(info.model.num_documents):
            nv = info.model._normalise_term_vector(lsi_plugin._C.T[d_i])
            fv = info.model._compute_document_feature_vector(nv)
            for fv_i in range(len(fv)):
                assert numpy.isclose(fv[fv_i], info.model.S_Vt_T[d_i][fv_i])

        # Verfiy documents match each other exactly
        for i, d_i in enumerate(doc_ids):
            r = lsi_plugin.compare_document(index, QueryStringQuery("id=doc-{}".format(d_i)))
            assert_almost_equal(max(r.values()), 1)
        similarities, sim_frame_ids = lsi_plugin.get_document_similarities()
        for i, d_similarities in enumerate(similarities):
            assert_almost_equal(d_similarities[i], 1)

        # Check compare document with model filter
        q = QueryStringQuery('environment')
        r = LSIPlugin(index).compare_document(index, QueryStringQuery("id=doc-{}".format(doc_ids[0])), q)
        assert len(r) == index.searcher().count(q)

        # Check document similarities with modified order
        f_ids = list(lsi_plugin._frame_ids)
        f_ids.reverse()
        assert lsi_plugin.get_document_similarities(f_ids)[1] == f_ids

        with pytest.raises(Exception):
            # Too many features
            index.run_plugin(LSIPlugin, num_features=index.get_vocab_size()+1)

        with pytest.raises(Exception):
            # Too many frames for comparison
            lsi_plugin.compare_document(index)
