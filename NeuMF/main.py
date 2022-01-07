import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

from NeuMF.neumf import GMFLayer, MLPModel, NeuMF
from NeuMF.movielens_layers import MovieModel, UserModel

# loading data
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
    "user_rating": x["user_rating"]
})

movies = movies.map(lambda x: x["movie_title"])

timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,
)

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_id"]))))

# preparing batched data
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# building model

# we first construct the embedding models for both
mf_user_model = UserModel(unique_user_ids, timestamp_buckets, 128, 128)
mf_movie_model = MovieModel(unique_movie_titles, movies, 128, 128)

mlp_user_model = UserModel(unique_user_ids, timestamp_buckets)
mlp_movie_model = MovieModel(unique_movie_titles, movies)

gmf_layer = GMFLayer(mf_user_model, mf_movie_model, 64)
mlp_layer = MLPModel(mlp_user_model, mlp_movie_model, [512, 256, 64])

neuMF = NeuMF(gmf_layer, mlp_layer)

neuMF.compile(optimizer=tf.keras.optimizers.Adam())

cached_train = train.shuffle(100_000).batch(8192)
cached_test = test.batch(4096)

neuMF.fit(cached_train, epochs=10)
















