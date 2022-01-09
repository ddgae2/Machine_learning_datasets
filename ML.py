# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import tensorflow as tf
import os
import seaborn as sns
#testing: david gae

def rr2mol():
	application_df[0].drop
	del application_df['url']
	return rr2mol 

def false():
	r1 = f1()
	false = []
	for i, j in enumerate(r1['data_channel_is_tech']):
		if j == 0:
		  false.append(i)
	return false

def true():
	true = []
	for i, j in enmuerate(r1['data_channel_is_tech']):
		if j == 1:
		   true.apend(i)
	return true 

def f1():
	list1 = []
	#create dataframe1 of heatmap
	for i,j in enumerate(application_df):
		list1.append(j)
		datafrm1 = pd.DataFrame(list1)
	return datafrm1

def heatmap():
	#heatmap
	ax = sns.heatmap(f1, linewidth=0.5)
	plt.show()



if __name__=="__main__":
	print(heatmap)
	application_df = pd.read_csv("./accel.csv")
	#del application_df['url']
	y = application_df.iloc[:,0]
	X = application_df
	#X = application_df.drop('year',axis=1)

# Split the preprocessed data into a training and testing dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
# Create a StandardScaler instances
	scaler = StandardScaler()

# Fit the StandardScaler
	X_scaler = scaler.fit(X_train)

# Scale the data
	X_train_scaled = X_scaler.transform(X_train)
	X_test_scaled = X_scaler.transform(X_test)
	X_train_scaled.shape

#deep neural
	number_input_features = len(X_train_scaled[0])
	nodes_hidden_layer1 = 80
	nodes_hidden_layer2 = 30

	nn = tf.keras.models.Sequential()

# First hidden layer
	nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer1, activation="relu", input_dim=number_input_features))

# Second hidden layer
	nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer2, activation="relu"))

# Output layer
	nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
	nn.summary()
	nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


import os
from tensorflow.keras.callbacks import ModelCheckpoint
# Define the checkpoint path and filenames
os.makedirs("checkpoints/", exist_ok=True)
checkpoint_path = "checkpoints/weights.{epoch:02d}hdf5"
# Train the model
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch',
    period=5)

fit_model = nn.fit(X_train_scaled, y_train, epochs=100, callbacks=[cp_callback])
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

