from keras import backend as K
from keras.engine.topology import Layer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from keras.initializers import Constant
from keras.layers import Reshape


class BoF_Pooling(Layer):
    """
    Implements the CBoF pooling
    """

    def __init__(self, n_codewords, spatial_level=0 ,**kwargs):
        """
        Initializes a BoF Pooling layer
        :param n_codewords: the number of the codewords to be used
        :param spatial_level: 0 -> no spatial pooling, 1 -> spatial pooling at level 1 (4 regions). Note that the
         codebook is shared between the different spatial regions
        :param kwargs:
        """
        self.N_k = n_codewords
        self.spatial_level = spatial_level
        self.V, self.sigmas = None, None
        super(BoF_Pooling, self).__init__(**kwargs)

    def build(self,input_shape):
        k = []
        for i in input_shape:
            if( i == None):
                k.append(i)
            else:
                k.append(int(i))
        input_shape = tuple(k)
        print(input_shape)
        self.V = self.add_weight(name='codebook', shape=(1, 1, input_shape[3], self.N_k), initializer='uniform',trainable=True)
        self.sigmas = self.add_weight(name='sigmas', shape=(1, 1, 1, self.N_k), initializer=Constant(0.1),
                                      trainable=True)
        super(BoF_Pooling, self).build(input_shape)

    def call(self, x):
        print("call")

        # Calculate the pairwise distances between the codewords and the feature vectors
        x_square = K.sum(x ** 2, axis=3, keepdims=True)
        y_square = K.sum(self.V ** 2, axis=2, keepdims=True)
        #print(K.conv2d(self.V,x, strides=(1, 1), padding='valid').shape)
        dists = x_square + y_square - 2 * K.conv2d(x,self.V , strides=(1, 1), padding='valid')
        dists = K.maximum(dists, 0)
        # Quantize the feature vectors
        quantized_features = K.softmax(- dists / (self.sigmas ** 2))

        # Compile the histogram
        if self.spatial_level == 0:
            histogram = K.mean(quantized_features, [1, 2])
        elif self.spatial_level == 1:
            shape = K.shape(quantized_features)
            mid_1 = K.cast(shape[1] / 2, 'int32')
            mid_2 = K.cast(shape[2] / 2, 'int32')
            histogram1 = K.mean(quantized_features[:, :mid_1, :mid_2, :], [1, 2])
            histogram2 = K.mean(quantized_features[:, mid_1:, :mid_2, :], [1, 2])
            histogram3 = K.mean(quantized_features[:, :mid_1, mid_2:, :], [1, 2])
            histogram4 = K.mean(quantized_features[:, mid_1:, mid_2:, :], [1, 2])
            histogram = K.stack([histogram1, histogram2, histogram3, histogram4], 1)
            histogram = K.reshape(histogram, (-1, 4 * self.N_k))
        else:
            # No other spatial level is currently supported (it is trivial to extend the code)
            assert False

        # Simple trick to avoid rescaling issues
        return histogram * self.N_k

    def compute_output_shape(self, input_shape):
        print("output")
        k = []
        for i in input_shape:
            if( i == None):
                k.append(i)
            else:
                k.append(int(i))
        input_shape = tuple(k)
        print(input_shape)
        if self.spatial_level == 0:
            return (input_shape[0], self.N_k)
        elif self.spatial_level == 1:
            return (input_shape[0], 4 * self.N_k)

'''
def initialize_bof_layers(model, data, n_samples=100, n_feature_samples=5000, batch_size=32, k_means_max_iters=300,
                          k_means_n_init=4):
    """
    Initializes the BoF layers of a keras model
    :param model: the keras model
    :param data: data to be used for initializing the model
    :param n_samples: number of data samples used for the initializes
    :param n_feature_samples: number of feature vectors to be used for the clustering process
    :param batch_size:
    :param k_means_max_iters: the maximum number of iterations for the clustering algorithm (k-means)
    :param k_means_n_init: defines how many times to run the k-means algorithm
    :return:
    """

    for i in range(len(model.layers)):
        if isinstance(model.layers[i], BoF_Pooling):
            print("Found BoF layer (layer %d), initializing..." % i)
            cur_layer = model.layers[i]

            # Compile a function for getting the feature vectors
            get_features = K.function([model.input] + [K.learning_phase()], [model.layers[i - 1].output])

            features = []
            for j in range(int(n_samples / batch_size)):
                cur_feats = get_features([data[j * batch_size:(j + 1) * batch_size], 0])[0]
                features.append(cur_feats.reshape((-1, cur_feats.shape[3])))
            features = np.concatenate(features)
            np.random.shuffle(features)
            features = features[:n_feature_samples]

            # Cluster the features
            kmeans = KMeans(n_clusters=cur_layer.N_k, n_init=k_means_n_init, max_iter=k_means_max_iters)
            kmeans.fit(features)
            V = kmeans.cluster_centers_.T
            V = V.reshape((1, 1, V.shape[0], V.shape[1]))

            # Set the value for the codebook
            K.set_value(cur_layer.V, np.float32(V))

            # Get the mean distance for initializing the sigmas
            mean_dist = np.mean(pairwise_distances(features[:100]))

            # Set the value for sigmas
            sigmas = np.ones((1, 1, 1, cur_layer.N_k)) * (mean_dist ** 2)
            K.set_value(cur_layer.sigmas, np.float32(sigmas))'''