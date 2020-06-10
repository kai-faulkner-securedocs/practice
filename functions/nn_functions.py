def get_data(path):
	import pandas as pd
	data = pd.read_csv(path, sep='\t')
	data = data[['review', 'rating']]
	return data

def clean_data(data):
	import pandas as pd
	import re
	import string
	data['review'] = data['review'].apply(lambda x: x.lower())
	data['review'] = data['review'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
	data['review'] = data['review'].apply(lambda x: re.sub('039', '', x))
	data['review'] = data['review'].apply(lambda x: re.sub('\r\n', '', x))
	data['rating'] = data['rating'].replace(range(8), 0).replace(range(8, 11), 1)
	return data

def split_data(data):
	import pandas as pd
	from tensorflow.keras.preprocessing.text import Tokenizer
	from tensorflow.keras.preprocessing.sequence import pad_sequences
	from tensorflow.keras.utils import to_categorical

	tokenizer = Tokenizer(num_words=5000, split=" ")
	tokenizer.fit_on_texts(data['review'].values)
	X = tokenizer.texts_to_sequences(data['review'].values)
	X = pad_sequences(X, maxlen=1659)
	y = to_categorical(data['rating'])
	return X, y

def build_model(dim):
	from tensorflow import keras
	from keras.models import Sequential
	from keras.layers import Embedding, Flatten, Dense, Dropout

	model = Sequential()
	model.add(Embedding(5000, 256, input_length=dim))
	model.add(Flatten())
	model.add(Dense(80, activation='relu'))
	model.add(Dense(70, activation='relu'))
	model.add(Dropout(.4))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(30, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	return model

def train_model(model, x, y, epoch=5, batch=32):
	from tensorflow import keras
	model.fit(x, y, epochs=epoch, batch_size=batch)
	return model

def evaluate_model(model, x, y):
	from tensorflow import keras
	return model.evaluate(x, y)