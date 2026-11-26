import subprocess
import data_processing

"""
Helper function that simply allows executing the OpenNMT commands via the Python code
The command and its arguments are passed as a list to the run() method (that is the reason for cmd.split())
"""
def execute_cmd(cmd):
    subprocess.run([t for t in cmd.split()], check=True)


"""
Pre-processing the data (splitting to train / validation sets, ensuring expected format)
"""
data_processing.process_training_data()
data_processing.process_test_data()


"""
Building the vocabulary
"""
execute_cmd('onmt_build_vocab -config a4-transformer.yaml -n_sample -1') # -1 means: build the vocabulary from the entire dataset

"""
Removing the counts from the vocabulary
"""

# For some reason, the token (in this case, character) counts were being included in the vocabulary and leading to predictions of only <unk>.
# To fix this, I will try to remove the counts from the vocabularies in-place, keeping only the symbols

VOCAB_SRC = 'data/vocab.src'
VOCAB_TGT = 'data/vocab.tgt'

def remove_counts(filename):
    chars = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.split():
                chars.append(line.split()[0])

    with open(filename, 'wt', encoding='utf-8', newline='\n') as f:
        for char in chars:
            f.write(char + '\n')

remove_counts(VOCAB_SRC)
remove_counts(VOCAB_TGT)


"""
Training the model
"""
execute_cmd('onmt_train -config a4-transformer.yaml')


"""
Making predictions using a model checkpoint, and saving them into a file
"""
# Model choice is based on the highest accuracy on the validation set (see training_logs.log)
# In this configuration, the total number of steps was 10,000
# I omitted the .pt file from the repository because it is large, but it can be retried locally if required
execute_cmd('onmt_translate -model data/run/model_step_6000.pt -src data/src-test.txt -output a4-transformer.predictions -gpu 0')

"""
Converting predictions to the expected format
"""

# Reading through the prediction outputted by the model (each line is a sequence of characters, separated by whitespace)
# Removing the whitespaces in the process (each prediction becomes a contiguous string of IPA symbols)
with open('a4-transformer.predictions', 'r', encoding='utf-8') as f:
    predictions = []
    for line in f:
        predictions.append(line.replace(' ', ''))

# Writing each prediction next to its original word from the test set. Filename remains the same.
with open('nlp2-a4.test', 'r', encoding='utf-8') as f:
    with open('a4-transformer.predictions', 'wt', encoding='utf-8') as p:
        idx = 0
        for word in f:
            word = word.strip()
            p.write(f'{word}\t{predictions[idx]}')
            idx += 1
