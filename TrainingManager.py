import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential

class TrainingManager_NN:
	def __init__(self, variables, training, epochs, name):
		np.random.seed(10)
		tf.random.set_seed(20)
		self.variables = variables
		self.training = training
		self.epochs = epochs
		self.name = name
		self.score = []
		self.score_test = []
		self.hist_acc = []
		self.hist_val_acc = []
		self.hist_loss = []
		self.hist_val_loss = []
		self.label = []
		self.model = None
		self.mean_acc = None
		self.mean_val_acc = None
		self.mean_val_loss = None
		self.mean_val_accuracy = None
		self.mean_test_loss = None
		self.mean_test_accuracy = None

	def train_model(self, train, label_train, validation, label_validation, test, label_test):
		print("-- TRAINING has started")

		np.random.seed(10)


		for i in range(self.training):
			np.random.seed(20)
			#tf_seed = tf.random.set_seed(i)
			tf_seed = tf.keras.utils.set_random_seed(i)
			tf.config.experimental.enable_op_determinism()
			print("Running training number {}/{}".format(i, self.training))

			self.model = self.build_model()

			self.model.compile(
				loss=tf.keras.losses.categorical_crossentropy,
				optimizer=Adam(use_ema=True),
				metrics=["categorical_accuracy"]
			)

			reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.0002, seed=tf_seed)
			callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)

			history = self.model.fit(
				train,
				label_train,
				callbacks=[reduce_lr],  # , callback],
				batch_size=512,
				epochs=self.epochs,
				shuffle=True,
				verbose=1,
				validation_data=(validation, label_validation)
			)

			self.score.append(self.model.evaluate(validation, label_validation, verbose=0))
			self.score_test.append(self.model.evaluate(test, label_test, verbose=1))
			self.hist_acc.append(history.history['categorical_accuracy'])
			self.hist_val_acc.append(history.history['val_categorical_accuracy'])
			self.hist_loss.append(history.history['loss'])
			self.hist_val_loss.append(history.history['val_loss'])
			
			self.model.save_weights(f"{self.name}/model_{self.name}.h.{i}")

			prediction = self.model.predict(test)
			self.label.append(prediction.tolist())
			
		self.compute_average_statistics()

	def compute_average_statistics(self):
		self.mean_acc = np.mean(self.hist_acc[:self.training], axis=0)
		self.mean_val_acc = np.mean(self.hist_val_acc[:self.training], axis=0)
		self.mean_loss = np.mean(self.hist_loss[:self.training], axis=0)
		self.mean_val_loss = np.mean(self.hist_val_loss[:self.training], axis=0)

		score_T = np.array(self.score).T
		score_test_T = np.array(self.score_test).T

		# Calcola statistiche medie
		self.mean_val_loss = float(sum(score_T[0])) / self.training
		self.mean_val_accuracy = float(sum(score_T[1])) / self.training
		self.mean_test_loss = float(sum(score_test_T[0])) / self.training
		self.mean_test_accuracy = float(sum(score_test_T[1])) / self.training

		# Stampa performance medie
		print('-- TRAINING RESULTS: ')
		print('Mean validation loss:', self.mean_val_loss)
		print('Mean validation accuracy:', self.mean_val_accuracy)
		print('Mean test loss:', self.mean_test_loss)
		print('Mean test accuracy:', self.mean_test_accuracy, '\n')

	def build_model(self):
		np.random.seed(10)
		self.model = Sequential()
		if self.variables == 5:
			self.model.add(Dense(50, input_shape=(5,), activation='relu'))  # input layer and first hidden layer
		if self.variables == 7:
			self.model.add(Dense(50, input_shape=(7,), activation='relu'))  # input layer and first hidden layer
		self.model.add(Dense(50, activation='relu'))  # second hidden layer
		self.model.add(Dense(30, activation='relu'))  # third hidden layer
		self.model.add(Dense(20, activation='relu'))  # fourth hidden layer
		self.model.add(Dense(2, activation='softmax'))  # softmax layer, used for classification (2 classes: signal and background)
		return self.model

