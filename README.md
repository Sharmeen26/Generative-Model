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
fasta_file_path = " "
# Load data from the FASTA file
sequences, virus_types = load_fasta_file(fasta_file_path)

# Print the number of sequences and types
num_sequences = len(sequences)
print("Number of Sequences:", num_sequences)

# Check the first few sequences and their associated virus types
for i in range(5):  # Adjust the range to see more sequences if needed
    print(f"Sequence {i+1}: {sequences[i]}")
    print(f"Virus Type: {virus_types[i]}")
    print("------")
