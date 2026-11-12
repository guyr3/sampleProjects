import torch
from sklearn.model_selection import train_test_split

TRAIN_DATASET = 'nlp2-a3.train'
TEST_DATASET = 'nlp2-a3.test'

"""
Reading the training dataset
Returns two lists:
    - X_train: contains each word without any whitespaces, as a list of characters
    - y_train: contains each word as a sequence of labels (B for beginning of a unit, I for inside a unit)
"""
def process_training_data(filename):
    X, y = [], []

    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            original = list(line.replace(' ', ''))
            target = ['B']

            for i in range(1, len(line)):
                if line[i] == ' ':
                    continue
                if line[i-1] == ' ':
                    target.append('B')
                else:
                    target.append('I')

            X.append(original)
            y.append(target)

    return X, y

def process_test_data(filename):
    X_test = []
    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            X_test.append(list(word)) # Each line in the test dataset contains a single word, simply split it to characters
    return X_test

X, y = process_training_data(TRAIN_DATASET)
X_test_original = process_test_data(TEST_DATASET)

"""
Creating a vocabulary of symbols & defining a mapping dictionary for characters and labels
"""
vocab = ['pad', 'unk'] + sorted(set([char for word in X for char in word])) # Handling padding & unknown words in the vocabulary
c2id = {vocab[i]:i for i in range(len(vocab))}

l2id = {'B': 0, 'I': 1}
B_ID = 0 # for reconstructing the output with whitespaces in the predicted locations

"""
Converting X and y to integer indices 
"""
X = [[c2id[char] for char in word] for word in X]
X_test = [[c2id.get(char, c2id['unk']) for char in word] for word in X_test_original] # In case of unfamiliar words in the test set, fallback to the unknown symbol idx

y = [[l2id[label] for label in sequence] for sequence in y]

"""
Padding X and y
"""
max_word = max([len(x) for x in X])

for i in range(len(X)):
    X[i] = X[i] + [0] * (max_word - len(X[i]))
X = torch.tensor(X, dtype=torch.long)

for i in range(len(X_test)):
    X_test[i] = X_test[i] + [0] * (max_word - len(X_test[i]))
X_test = torch.tensor(X_test, dtype=torch.long)

# For the labels, I use a padding value of -100, which CrossEntropyLoss ignores by default
for i in range(len(y)):
    y[i] = y[i] + [-100] * (max_word - len(y[i]))
y = torch.tensor(y, dtype=torch.long)

"""
Defining the model (Bi-directional LSTM) 
"""

class BiLSTM(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64):
        super(BiLSTM, self).__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.ff = torch.nn.Linear(hidden_dim * 2, 2) # using hidden_dim * 2 because of bi-directionality, then 2 possible outputs B or I

    def forward(self, x):
        e = self.emb(x)
        r, _ = self.lstm(e)
        out = self.ff(r)
        return out


"""
The training loop
"""

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1) # splitting training set to training & validation sets

bi_lstm = BiLSTM(len(vocab))
lossfn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(bi_lstm.parameters(), lr=0.01)
batch_size = 10

num_epochs = 200

best_val_loss = float("inf")

for epoch in range(num_epochs):
    batch_start = 0
    epoch_loss = 0

    while batch_start < X_train.shape[0]:
        batch_x = X_train[batch_start:batch_start+batch_size]
        batch_y = y_train[batch_start:batch_start+batch_size]
        batch_start += batch_size

        optimizer.zero_grad()
        pred = bi_lstm.forward(batch_x)
        loss = lossfn(pred.permute(0,2,1), batch_y) # permuting to make output shape match expected shape for CrossEntropyLoss
        epoch_loss += loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, loss {epoch_loss / (X_train.shape[0] // batch_size)}")

    # Compute loss on validation set
    with torch.no_grad():
        val_pred = bi_lstm.forward(X_validation)
        val_loss = lossfn(val_pred.permute(0, 2, 1), y_validation)
        print(f'Epoch {epoch}, val loss: {val_loss}')

        # Save the model with the lowest validation set loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(bi_lstm.state_dict(), 'best_model.pt')


"""
Making predictions
"""
bi_lstm.load_state_dict(torch.load('best_model.pt'))
bi_lstm.eval()

with torch.no_grad():
    logit_predictions = bi_lstm(X_test) # The model gives us two logits (for B and for I) per state in the sequence

padded_predictions = torch.argmax(logit_predictions, dim=2) # We take the higher logit as the prediction for each state

predictions = []
for i in range(len(X_test_original)):
    prediction = ""
    original_word = X_test_original[i]
    label_sequence = padded_predictions[i][:len(original_word)] # The predictions are padded to max_word length, we only want up to original word length

    for c in range(len(original_word)):
        if c > 0 and label_sequence[c] == B_ID:
            prediction += ' '
        prediction += original_word[c]

    predictions.append(prediction)

# Writing the predictions into a file
with open('a3-BiLSTM.predictions', 'wt', encoding='utf-8') as f:
    for prediction in predictions:
        print(prediction, file=f)