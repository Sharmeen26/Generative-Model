# Step 1: Load Data
from Bio import SeqIO


# Function to load data from a FASTA file
def load_fasta_file(file_path):
    sequences = []
    virus_types = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        virus_type = record.description.split()[
            0
        ]  # Assuming virus type is the first element
        virus_types.append(virus_type)
    return sequences, virus_types


# Path to your FASTA file
fasta_file_path = "file path"
# Load data from the FASTA file
sequences, virus_types = load_fasta_file(fasta_file_path)

# Print the number of sequences and types
num_sequences = len(sequences)
print("Number of Sequences:", num_sequences)


# Step 2 Prepare Data For Model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def prepare_data(sequences):
    all_sequences = sequences
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(all_sequences)

    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for seq in all_sequences:
        token_list = tokenizer.texts_to_sequences([seq])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(seq) for seq in input_sequences])
    padded_sequences = pad_sequences(
        input_sequences, maxlen=max_sequence_length, padding="pre"
    )

    predictors, label = padded_sequences[:, :-1], padded_sequences[:, -1]

    return predictors, label, total_words, max_sequence_length, tokenizer
    # Prepare data
predictors, label, total_words, max_sequence_length, tokenizer = prepare_data(sequences)

# Check the prepared data
print(predictors.shape)
print(label.shape)
print(total_words)
print(max_sequence_length)

# Check the first few sequences and their associated virus types
for i in range(5):  # Adjust the range to see more sequences if needed
    print(f"Sequence {i+1}: {sequences[i]}")
    print(f"Virus Type: {virus_types[i]}")
    print("------")

    # Step 3 Sequence predicting framework
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Example values for total_words and max_sequence_length
total_words = 10000
max_sequence_length = 100

def build_model(total_words, max_sequence_length):
    model = Sequential()
    model.add(Embedding(input_dim=total_words, output_dim=50, input_length=max_sequence_length - 1))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model

# Build the model
model = build_model(total_words, max_sequence_length)

# Trigger model building by passing an example input
model.build(input_shape=(None, max_sequence_length - 1))

# Print model summary
model.summary()

#Step 4: Train The Model
from tensorflow.keras.callbacks import EarlyStopping
# Adjust epochs,verbose and patience according to your data
def train_model(model, predictors, label):
    model.fit(predictors, label, epochs=15, verbose=1, callbacks=[EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)])

# Train the model
train_model(model, predictors, label)
#Step 4: Train The Model
from tensorflow.keras.callbacks import EarlyStopping
# Adjust epochs,verbose and patience according to your data
def train_model(model, predictors, label):
    model.fit(predictors, label, epochs=15, verbose=1, callbacks=[EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)])

# Train the model
train_model(model, predictors, label)


# Step 5: Generating New Sequence Type
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def generate_sequence(model, tokenizer, max_sequence_length, seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_length - 1, padding="pre"
        )
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Choose the word with some randomness to introduce diversity
        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        seed_text += output_word.upper()

    return seed_text


def get_user_input():
    while True:
        generate_sequence_input = input(
            "Do you want to generate a sequence? (yes/no): "
            ).lower()
        if generate_sequence_input == "yes":
            length_input = int(
                input(
                    "Enter the length of the sequence to generate (between 100 and 15000): "
                )
            )
            if 100 <= length_input <= 15000:
                num_sequences_input = int(
                    input(
                        "Enter the number of sequences to generate (between 1 and 10): "
                    )
                )
                if 1 <= num_sequences_input <= 10:
                    return True, length_input, num_sequences_input
                else:
                    print(
                        "Please enter a number between 1 and 10 for the number of sequences."
                    )
            else:
                print("Please enter a number between 100 and 15000.")
        elif generate_sequence_input == "no":
            return False, 0, 0
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
while True:
    generate, length, num_sequences = get_user_input()
    if not generate:
        break

    seed_sequence = "..."  # You can change this to start with any type you want
    for _ in range(num_sequences):
        generated_sequence = generate_sequence(
            model, tokenizer, max_sequence_length, seed_sequence, length
        )

        # Print the generated sequence
        print(generated_sequence)


    
