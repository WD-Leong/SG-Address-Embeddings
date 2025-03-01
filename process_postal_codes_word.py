# Import the libraries. #
import pandas as pd
import pickle as pkl
from nltk import wordpunct_tokenize

# Load the raw data. #
postal_codes = "../Data/postal_codes/"
postal_codes += "sg_postal_codes_shorthand.csv"

tmp_postal_cd = pd.read_csv(postal_codes)
tmp_postal_cd = tmp_postal_cd[tmp_postal_cd["POSTAL"] > 0]
postal_codes  = list(sorted(
    list(set(list(tmp_postal_cd["POSTAL"].values)))))
tmp_addresses = list(set(list(tmp_postal_cd["ADDRESS"].values)))

# Generate the postal code dictionary. #
postal_codes = [str(x) for x in postal_codes]
post_cd_2_idx = dict([
    (postal_codes[x], x) for x in range(len(postal_codes))])
idx_2_post_cd = dict([
    (x, postal_codes[x]) for x in range(len(postal_codes))])
print("Total of", len(post_cd_2_idx), "unique postal codes.")

# Get the word token dictionary. #
all_tokens = []
for tmp_address in tmp_addresses:
    tmp_tokens = wordpunct_tokenize(
        str(tmp_address).lower())[:-1]
    all_tokens = list(set(all_tokens + tmp_tokens))

all_tokens = list(sorted(list(set(all_tokens))))
all_tokens = ["PAD", "SOS", "EOS"] + all_tokens

word_2_idx = dict([
    (all_tokens[x], x) for x in range(len(all_tokens))])
idx_2_word = dict([
    (x, all_tokens[x]) for x in range(len(all_tokens))])
print("Total of", len(word_2_idx), "word tokens.")

# Generate the word index. #
tmp_tuple  = []
seq_length = []

for tmp_address in tmp_addresses:
    tmp_tokens = wordpunct_tokenize(
        str(tmp_address).lower().strip())
    
    # Addresses always end with Singapore <Postal Code>. #
    token_idx  = [
        word_2_idx[x] for x in tmp_tokens[:-2]]
    postal_idx = post_cd_2_idx[str(int(tmp_tokens[-1]))]
    
    seq_length.append(len(token_idx))
    tmp_tuple.append((token_idx, postal_idx))

n_seq_len = max(seq_length)
print("Maximum sequence length of", n_seq_len, "tokens.")

tmp_pkl_file = "../Data/postal_codes/"
tmp_pkl_file += "postal_codes_word.pkl"
with open(tmp_pkl_file, "wb") as tmp_save:
    pkl.dump(tmp_tuple, tmp_save)
    pkl.dump(word_2_idx, tmp_save)
    pkl.dump(idx_2_word, tmp_save)
    pkl.dump(post_cd_2_idx, tmp_save)
    pkl.dump(idx_2_post_cd, tmp_save)
print("Processing completed.")
