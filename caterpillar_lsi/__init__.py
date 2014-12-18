# Copyright (C) Kapiche
# Author: Kris Rogers <kris@kapiche.com>
# License: GNU Affero General Public License
"""
Caterpillar latent semantic indexing plugin.

The ``LSIPlugin`` generates an ``LSIModel`` for index data which facilitates the comparison of existing or new documents
with documents in the index. All references to document entities assume that each document has been indexed as a single
frame.

"""
from __future__ import division

from collections import defaultdict
import math
import pickle

from caterpillar.processing.plugin import AnalyticsPlugin
from caterpillar.storage import ContainerNotFoundError
import numpy
from scipy import linalg, spatial
import ujson as json


class LSIModel(object):
    """
    Provides generation and manipulation of LSI models.

    Arguments:
    C -- term x document frequency matrix.
    num_features -- number of 'features' to consider.
    normalise_frequencies -- True to normalise frequencies in ``C`` according to tf-idf.
    calculate_document_similarities -- True to calculate an adjacency matrix of all document-document similarities.

    """
    def __init__(self, C, num_features, normalise_frequencies, calculate_document_similarities):
        C = numpy.array(C, copy=True)
        self.num_features = num_features
        self.normalise_frequencies = normalise_frequencies
        self.num_terms, self.num_documents = C.shape

        # Optional frequency normalisation
        if normalise_frequencies is True:
            C = C.astype(numpy.float)
            self.idfs = []
            for ti in range(self.num_terms):
                idf = math.log(self.num_documents / numpy.count_nonzero(C[ti]))
                self.idfs.append(idf)
                C[ti] *= idf

        # Perform SVD
        # U - term x term_truncated matrix
        # S - feature x feature matrix
        # Vt - document_truncated x document matrix
        U, s, Vt = linalg.svd(C, full_matrices=False)

        # Convert feature vector to diagonal matrix
        S = numpy.diag(s)

        # Dimension reduction
        self.U = U[:, :num_features]  # New shape should be (num_terms, num_features)
        self.S = S[:num_features, :num_features]  # New shape should be (num_features, num_features)
        Vt = Vt[:num_features]  # New shape should be (num_features, num_documents)

        # Compute lower dimensional document-feature matrix (for comparing documents)
        # ``self.S_Vt_T[x]`` provides feature vector for document 'x'
        self.S_Vt_T = numpy.dot(self.S, Vt).T

        if calculate_document_similarities:
            # Compute document-document similarities
            document_similarities = defaultdict(dict)
            for i in xrange(self.num_documents):
                dfv = self.S_Vt_T[i]
                for j in xrange(i, self.num_documents):
                    distance = spatial.distance.cosine(dfv, self.S_Vt_T[j])
                    similarity = 1 - distance
                    document_similarities[i][j] = document_similarities[j][i] = similarity
            self.document_similarities = [d.values() for d in document_similarities.itervalues()]

    def compare_document(self, term_frequencies, filter_docs=None):
        """
        Determine similarity of a new document with all documents in this model.

        Required Arguments:
        term_frequencies -- A vector of term frequencies for the new document.
        filter_docs -- A list of document indices to consider when comparing. Defaults to None (all documents).

        Returns a dict containing doc_id -> similarity.

        """
        if len(term_frequencies) != self.num_terms:
            raise LSIModel.BadDocumentException("Document vector length ({}) is not equal to num terms ({})"
                                                .format(len(term_frequencies), self.num_terms))

        # Convert vector to m x 1 matrix
        term_frequencies = numpy.array(term_frequencies).T

        if numpy.count_nonzero(term_frequencies) == 0:
            raise LSIModel.EmptyDocumentException(
                "Document is empty. This can occur when a document does not share any terms with the model.")

        # Normalise new document if model has been constructed from normalised data
        if self.normalise_frequencies:
            term_frequencies = self._normalise_term_vector(term_frequencies)

        # Compute document-feature vector
        dfv = self._compute_document_feature_vector(term_frequencies)

        # Compute similarity with document-feature vectors in the model
        similarities = {}
        for d_i in xrange(len(self.S_Vt_T)):
            if filter_docs and d_i not in filter_docs:
                continue
            other_dfv = self.S_Vt_T[d_i]
            distance = spatial.distance.cosine(dfv, other_dfv)
            similarity = 1 - distance
            similarities[d_i] = max(0, similarity)  # Prevent negative floating point errors

        return similarities

    @staticmethod
    def loads(model_str):
        """
        Load string generated by ``dumps`` into a new ``LSIModel`` object.

        """
        return pickle.loads(model_str)

    def dumps(self):
        """
        Dump this ``LSIModel`` to a string that can be used by the ``loads`` method to regenerate the model.

        """
        return pickle.dumps(self)

    def _compute_document_feature_vector(self, t_v):
        """
        Compute the feature vector for a document, specified by its term frequency vector ``t_v`` (m x 1).

        The returned vector is equivalent to a new row in ``self.S_VT_T``.

        This 'folding-in' process effectively coerces a new document into the same lower-dimensional feature space
        as our existing model, in turn facilitating direct comparison.

        """
        # Multiply term frequency vector for new document by term weights in existing LSI model
        t_v = numpy.dot(t_v, self.U)
        # Multiply by S^-1 to project document into the lower-dimensional feature space
        f_v = numpy.dot(t_v, linalg.inv(self.S))
        # At this stage, f_v is equivalent to a new column in Vt
        # Multiply it by S to compute the documents feature vector for comparison
        return numpy.dot(f_v, self.S)

    def _normalise_term_vector(self, t_v):
        """
        Normalise the specified term frequency vector (m x 1) using TF-IDF.

        The IDF values are calculated when creating a normalised LSI model.

        """
        return t_v * self.idfs

    class BadDocumentException(Exception):
        """Incorrect document format/structure"""
        pass

    class EmptyDocumentException(Exception):
        """Document has no values"""
        pass


class LSIPluginInfo(object):
    """
    Encapsulates information about the running of ``LSIPlugin``.

    """
    def __init__(self, model, frame_ids, term_values):
        self.model = model
        self.frame_ids = frame_ids
        self.term_values = term_values


class LSIPlugin(AnalyticsPlugin):
    """
    An ``AnalyticsPlugin`` which provides Latent Semantic Indexing capabilities. This is primarily of interest for
    comparing and classifying documents.

    """
    NAME = 'lsi'
    INFO_CONTAINER = 'info'
    INFO_FRAMES = 'info_frames'
    INFO_MODEL = 'info_model'
    INFO_TERMS = 'info_terms'

    def __init__(self, reader):
        self._reader = reader
        try:
            # Attempt to intialise this plugin from storage
            info = self.get_info()
            self._model = info.model
            self._frame_ids = info.frame_ids
            self._term_values = info.term_values
        except ContainerNotFoundError:
            # No pre-existing plugin results available
            self._model = None
        super(LSIPlugin, self).__init__(reader)

    def get_name(self):
        return LSIPlugin.NAME

    def run(self, num_features=300, normalise_frequencies=True, calculate_document_similarities=False):
        """
        Run the plugin, extracting ``num_features`` latent features. If ``normalise_frequencies`` is Truth-y,
        frequencies will be normalised according to tf-idf before extracting features, which is generally recommended.

        Additionally, if ``calculate_document_similarities`` is Truth-y, this plugin will generate an adjacency matrix
        containing similarities between all document pairs in the model, accessible via ``get_document_similarities``.

        """
        # First build covariance matrix (frequency -- terms x frames)
        self._C, self._term_values, self._frame_ids = LSIPlugin._build_covariance_matrix(self._reader)

        num_rows = self._C.shape[0]
        if num_features > num_rows:
            raise RuntimeError("Number of features must be less than or equal to number of terms.")

        self._model = LSIModel(self._C, num_features, normalise_frequencies, calculate_document_similarities)

        return {
            LSIPlugin.INFO_CONTAINER: {
                LSIPlugin.INFO_FRAMES: self._frame_ids,
                LSIPlugin.INFO_MODEL: self._model.dumps(),
                LSIPlugin.INFO_TERMS: self._term_values
            }
        }

    def compare_index(self, reader, model_filter_query=None):
        """
        Compare all the documents in ``index`` to all documents in the model generated by the running of this plugin.
        A document is considered to be a frame on the index.

        Specifying ``model_filter_query`` allows filtering of the LSI model to compare against only a subset.

        This method returns a dict of similarities as follows:

            {
                frame_id: {
                    doc_id: similarity
                }
            }

        """
        if self._model is None:
            raise RuntimeError('Cannot compare index with model before the plugin has been run.')

        # Determine which docs to consider in the model
        filter_docs = None
        if model_filter_query is not None:
            frame_ids = self._reader.searcher().filter(model_filter_query)
            filter_docs = []
            for f_id in frame_ids:
                filter_docs.append(self._frame_ids.index(f_id))

        D, term_values, frame_ids = LSIPlugin._build_covariance_matrix(reader, self._term_values)
        order = [f_id for f_id in reader.get_frame_ids()]
        similarities = {frame_id: {} for frame_id in order}
        for i, term_freqs in enumerate(D.T):
            results = self._model.compare_document(term_freqs, filter_docs=filter_docs)
            for doc_id in results.keys():
                f_id = self._frame_ids[doc_id]
                results[f_id] = results.pop(doc_id)
            similarities[order[i]] = results
        return similarities

    def get_document_similarities(self, order_frame_ids=None):
        """
        This method returns a matrix of all document-document similarities.

        Only available when this plugin is run with a True value for ``calculate_document_similarities``.

        Optionally order according to ``order_frame_ids``.

        """
        if not hasattr(self._model, 'document_similarities'):
            raise RuntimeError('Document similarities are only available if their calculation is enabled at runtime')

        if order_frame_ids is None:
            return (self._model.document_similarities, self._frame_ids)

        item_order = [self._frame_ids.index(frame_id) for frame_id in order_frame_ids]
        similarities = [self._model.document_similarities[i] for i in item_order]
        for row in similarities:
            row[:] = [row[i] for i in item_order]
        return (similarities, order_frame_ids)

    def get_info(self):
        """
        Return ``LSIPluginInfo`` object for the running of this plugin.

        """
        if hasattr(self, '_model') and self._model is not None:
            return LSIPluginInfo(self._model, self._frame_ids, self._term_values)
        info = {k: v for k, v in self._reader.get_plugin_data(self, LSIPlugin.INFO_CONTAINER)}
        return LSIPluginInfo(LSIModel.loads(json.loads(info[LSIPlugin.INFO_MODEL])),
                             json.loads(info[LSIPlugin.INFO_FRAMES]), json.loads(info[LSIPlugin.INFO_TERMS]))

    @staticmethod
    def _build_covariance_matrix(index, row_terms=None):
        """
        Build a covariance matrix (term x document frequency) for the specified index. If ``row_terms`` is specifed,
        use them exclusively for the row dimension of the matrix. If none of the specified terms are present in the
        index, the covariance matrix will contain only zeros.

        Returns a 3-tuple containing (covariance_matrix, term_values, frame_ids).

        """
        positions = {k: v for k, v in index.get_positions_index()}
        row_terms = row_terms or positions.keys()  # rows
        frame_ids = [f_id for f_id in index.get_frame_ids()]  # columns
        term_frame_frequencies = []  # values
        for term in row_terms:
            frame_positions = positions.get(term, {})
            term_frame_frequencies.append([len(frame_positions.get(frame_id, [])) for frame_id in frame_ids])
        return (numpy.array(term_frame_frequencies), row_terms, frame_ids)
