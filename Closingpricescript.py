from spreadsheetfunctions import *
joineddf = pd.read_pickle('joineddf.pkl')
onlynorthformatted= pd.read_pickle("onlynorthformatted.pkl")

dataset = joineddf.values
split= int(.9*len(onlynorthformatted))
train = dataset[0:split,:]
valid = dataset[split:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create a data structure with 60 timesteps and 1 output
x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping (batch_size, timesteps, input_dim)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


#predicting values, using past 60 from the train data
inputs = joineddf[len(joineddf) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

#converts series matrix like
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

path_to_dataset = 'joineddf.csv'
sequence_length = 30

vector_vix = []
with open(path_to_dataset) as f:
    next(f) # skip the header row
    for line in f:
        fields = line.split(',')
        vector_vix.append(float(fields[1]))

matrix_vix = convertSeriesToMatrix(vector_vix, sequence_length)

matrix_vix = np.array(matrix_vix)
shifted_value = matrix_vix.mean()
matrix_vix -= shifted_value
print("Data  shape: ", matrix_vix.shape)

train_row = int(round(0.8 * matrix_vix.shape[0]))
train_set = matrix_vix[:train_row, :]

np.random.seed(1234)

np.random.shuffle(train_set)
# the training set
X_train = train_set[:, :-1]
# the last column is the true value to compute the mean-squared-error loss
y_train = train_set[:, -1]
# the test set
X_test = matrix_vix[train_row:, :-1]
y_test = matrix_vix[train_row:, -1]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model= load_model('RNN-erot.h5')

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

closing_price = model.predict(X_test)

pickle_out = open("closing_price.pickle","wb")
pickle.dump(closing_price, pickle_out)
pickle_out.close()
