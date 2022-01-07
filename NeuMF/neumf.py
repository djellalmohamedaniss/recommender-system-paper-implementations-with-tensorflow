from typing import Dict, Text

import tensorflow as tf
import tensorflow_recommenders as tfrs


class GMFLayer(tf.keras.layers.Layer):

    def __init__(self, user_model: tf.keras.Model, movie_model: tf.keras.Model, output_size=128):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model

        self.dense = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        user_embeddings = self.user_model(inputs)
        movies_embeddings = self.movie_model(inputs)
        element_wise_embeddings = tf.multiply(user_embeddings, movies_embeddings)
        outputs = self.dense(element_wise_embeddings)
        return outputs


class MLPModel(tf.keras.Model):

    def __init__(self, user_model: tf.keras.Model, item_model: tf.keras.Model, layer_sizes):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.dense_layers = tf.keras.Sequential()

        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        user_embeddings = self.user_model(inputs)
        item_embeddings = self.item_model(inputs)
        concatenated_embedding = tf.concat([user_embeddings, item_embeddings], -1)
        return self.dense_layers(concatenated_embedding)


class NeuMF(tfrs.models.Model):

    def __init__(self, mlp_model: MLPModel, gmf_model: GMFLayer, layer_size=128):
        super().__init__()
        self.mlp_model = mlp_model
        self.gmf_model = gmf_model
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(layer_size),
            tf.keras.layers.Dense(1)
        ])
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        mlp_result = self.mlp_model(features)
        gmf_model = self.gmf_model(features)
        concatenated_result = tf.concat([mlp_result, gmf_model], -1)
        return self.ratings(concatenated_result)

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("user_rating")
        rating_predictions = self(features)
        return self.task(labels=labels, predictions=rating_predictions)