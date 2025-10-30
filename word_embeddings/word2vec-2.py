import gensim.downloader as api
import pickle
from pathlib import Path

# Cache the model to avoid re-downloading on every run
cache_dir = Path('model_cache')
cache_dir.mkdir(exist_ok=True)
model_cache_path = cache_dir / 'word2vec-google-news-300.pkl'

if model_cache_path.exists():
    print("Loading Word2Vec model from cache...")
    with open(model_cache_path, 'rb') as f:
        wv = pickle.load(f)
else:
    print("Downloading Word2Vec model... (this may take a while on first run)")
    wv = api.load('word2vec-google-news-300')
    print("Caching model for future use...")
    with open(model_cache_path, 'wb') as f:
        pickle.dump(wv, f)




for index, word in enumerate(wv.index_to_key):
    if index == 20:
        break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")


vec_life = wv['life']
print(vec_life[:10])

try:
    vec_wretched = wv['wretched']
except KeyError:
    print("The word 'wretched' does not appear in this model")


pairs = [
    ('communism', 'car'),   # a minivan is a kind of car
    ('communism', 'tea'),   # still a wheeled vehicle
    ('communism', 'date'),  # ok, no wheels, but still a vehicle
    ('communism', 'life'),    # ... and so on
    ('communism', 'equality'),
]
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))


print(wv.most_similar(positive=['communism', 'date'], topn=5))

print(wv.doesnt_match(['communism', 'car', 'date', 'life', 'tee', 'equality']))
#Output:
# word #0/3000000 is </s>
# word #1/3000000 is in
# word #2/3000000 is for
# word #3/3000000 is that
# word #4/3000000 is is
# word #5/3000000 is on
# word #6/3000000 is ##
# word #7/3000000 is The
# word #8/3000000 is with
# word #9/3000000 is said
# word #10/3000000 is was
# word #11/3000000 is the
# word #12/3000000 is at
# word #13/3000000 is not
# word #14/3000000 is as
# word #15/3000000 is it
# word #16/3000000 is be
# word #17/3000000 is from
# word #18/3000000 is by
# word #19/3000000 is are
# [-0.06787109  0.09521484  0.03564453  0.171875    0.203125    0.04492188
#   0.125      -0.06445312  0.17675781  0.06201172]
# 'communism'	'car'	0.06
# 'communism'	'tea'	0.10
# 'communism'	'date'	0.03
# 'communism'	'life'	0.20
# 'communism'	'equality'	0.27
# [('Communism', 0.6643747687339783), ('Soviet_communism', 0.542768657207489), ('communist_regimes', 0.5314956903457642), ('Soviet_Communism', 0.5281749963760376), ('Communist_regimes', 0.5142533779144287)]
# date
