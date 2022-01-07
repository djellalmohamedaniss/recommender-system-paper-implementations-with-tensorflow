import tensorflow as tf


class UserModel(tf.keras.Model):

    def __init__(self, unique_user_ids, timestamp_buckets, user_embed_size=64, time_embed_size=64):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, user_embed_size),
        ])
        self.timestamp_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 1, time_embed_size),
        ])

    def call(self, inputs):
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.timestamp_embedding(inputs["timestamp"])
        ], axis=1)


class MovieModel(tf.keras.Model):

    def __init__(self, unique_movie_titles, movies, title_embed_size=64, title_text_embed_size=64):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, title_embed_size)
        ])

        self.title_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens)

        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, title_text_embed_size, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(movies)

    def call(self, features):
        return tf.concat([
            self.title_embedding(features["movie_title"]),
            self.title_text_embedding(features["movie_title"]),
        ], axis=1)
