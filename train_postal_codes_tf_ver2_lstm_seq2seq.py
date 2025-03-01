# Import the libraries. #
import time
import numpy as np
import pandas as pd
import pickle as pkl
#import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_lstm_scan as tf_lstm
import tf_ver2_lstm_ffwd_scan as tf_ffwd_lstm

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    encoder, decoder, optimizer, sub_batch_sz, 
    x_in_seq, x_out_seq, learning_rate=1.0e-3, gradient_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_in_seq.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = [x for x in encoder.trainable_variables]
    model_params += [x for x in decoder.trainable_variables]
    acc_gradients = [tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = x_in_seq[id_st:id_en, :]
        tmp_decode = x_out_seq[id_st:id_en, :-1]
        dec_labels = x_out_seq[id_st:id_en, 1:]
        
        with tf.GradientTape() as grad_tape:
            # Encoder. #
            encode_state = encoder.decode(
                tmp_encode, return_states=True, training=True)
            
            # Decoder Loss. #
            tmp_losses = decoder.decode_xent_loss(
                tmp_decode, dec_labels, 
                c_initial=encode_state, training=True)
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clip_tuple = tf.clip_by_global_norm(
        acc_gradients, gradient_clip)
    optimizer.apply_gradients(
        zip(clip_tuple[0], model_params))
    return avg_losses

# Model Parameters. #
batch_size = 128
sub_batch  = 128
seq_length = 25
num_layers = 3
num_rounds = 3

gradient_clip = 1.00
maximum_iter  = 3000
restore_flag  = False
save_step     = 250
warmup_steps  = 5000
display_step  = 50
anneal_step   = 2500
anneal_rate   = 0.75

prob_keep = 0.9
hidden_size = 256
warmup_flag = True
cooling_step = 250

model_ckpt_dir  = "../TF_Models/lstm_seq2seq_keras_postal_codes_3_layers"
train_loss_file = "train_loss_lstm_seq2seq_keras_postal_codes_3_layers.csv"

# Load the data. #
tmp_pkl_file = "../Data/postal_codes/"
tmp_pkl_file += "postal_codes_word.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    postal_codes = pkl.load(tmp_load_file)

    word_2_idx = pkl.load(tmp_load_file)
    idx_2_word = pkl.load(tmp_load_file)
    post_cd_2_idx = pkl.load(tmp_load_file)
    idx_2_post_cd = pkl.load(tmp_load_file)

vocab_size = len(word_2_idx)
print("Vocabulary Size:", str(vocab_size), "tokens.")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

tmp_data = []
for tmp_row in postal_codes:
    if len(tmp_row) > 1 and \
        len(tmp_row) <= seq_length:
        tmp_data.append(tmp_row)

# No UNK token in this dataset. #
num_data  = len(tmp_data)
SOS_token = word_2_idx["SOS"]
EOS_token = word_2_idx["EOS"]
PAD_token = word_2_idx["PAD"]
#UNK_token = word_2_idx["UNK"]
print("Total of", str(len(tmp_data)), "rows loaded.")

# Build the LSTM. #
print("Building the LSTM Seq2Seq Keras Model.")
start_time = time.time()

lstm_enc_model = tf_lstm.LSTM(
    num_layers, hidden_size, vocab_size, 
    seq_length+2, rate=1.0-prob_keep, res_conn=True)

lstm_dec_model = tf_ffwd_lstm.LSTM(
    num_layers, hidden_size, vocab_size, 
    seq_length+1, rate=1.0-prob_keep, res_conn=True)
lstm_optimizer = tfa.optimizers.AdamW(weight_decay=1.0e-4)

elapsed_time = (time.time() - start_time) / 60
print("LSTM Seq2Seq Keras Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    lstm_enc_model=lstm_enc_model, 
    lstm_dec_model=lstm_dec_model, 
    lstm_optimizer=lstm_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(
            manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Train the LSTM model. #
tmp_in_seq  = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_out_seq = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)

tmp_test_in  = np.zeros([1, seq_length+2], dtype=np.int32)
tmp_test_sos = SOS_token + np.zeros([1, 1], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.001
    learning_rate = max(
        anneal_rate**(n_iter // anneal_step)*initial_lr, 1.0e-5)

print("-" * 50)
print("Training the LSTM Model", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            anneal_factor = np.power(
                anneal_rate, int(n_iter / anneal_step))
            learning_rate = \
                max(anneal_factor*initial_lr, 1.0e-6)
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_in_seq[:, :]  = PAD_token
    tmp_out_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_p_idx = [SOS_token]
        tmp_p_idx += tmp_data[tmp_index][0] + [EOS_token]
        
        n_input = len(tmp_p_idx)
        tmp_in_seq[n_index, -n_input:] = tmp_p_idx
        tmp_out_seq[n_index, :n_input] = tmp_p_idx
        del tmp_p_idx
    
    # Compute the loss. #
    tmp_loss = sub_batch_train_step(
        lstm_enc_model, lstm_dec_model, lstm_optimizer, 
        sub_batch, tmp_in_seq, tmp_out_seq, learning_rate=learning_rate)

    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        tmp_test_in[:, :] = PAD_token
        sample_test = np.random.choice(num_data, size=1)
        tmp_p_index = tmp_data[sample_test[0]][0]
        
        in_phrase = " ".join(
            [idx_2_word[x] for x in tmp_p_index])
        n_tokens  = len(tmp_p_index)
        tmp_test_in[0, -n_tokens:] = tmp_p_index
        
        # Encoder. #
        tmp_state = lstm_enc_model.decode(
            tmp_test_in, training=False)
        
        # Decoder. #
        tmp_infer = lstm_dec_model.infer(
            tmp_test_sos, c_initial=tmp_state, 
            gen_len=seq_length, sample=False)
        del sample_test, n_tokens
        
        gen_phrase = " ".join([
            idx_2_word[x] for x in tmp_infer[0].numpy()])
        del tmp_p_index
        
        print("Iteration", str(n_iter)+".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip)+".")
        print("Learning Rate:", str(learning_rate)+".")
        print("Average Loss:", str(avg_loss)+".")
        
        print("")
        print("Input Phrase:")
        print(in_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        
        train_loss_list.append((n_iter, avg_loss))
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)
