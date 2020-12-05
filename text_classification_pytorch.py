# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:51:13 2020

@author: Alex
"""

# Writing Script
import argparse
import random
import sys
import time
import os
from functools import partial

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import (DataLoader,
                              TensorDataset,
                              RandomSampler,
                              SequentialSampled)
from transformers import AutoModelForSequenceClassification as AutoModel
from transformers import (AutoTokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup)

def read_file(filename, **kwards):
  """
  Read in a file
  """
  if filename.endswith('.xlsx'):
    return pd.read_excel(filename, **kwargs)
  elif filename.endswith('.csv'):
    return pd.read_csv(filename, **kwargs)
  raise ValueError('File must be CSV or Excel')

def write_file(df, filename, **kwargs):
  """
  Write out a file
  """
  if 'index' not in kwargs:
    kwargs['index'] = False
  if filename.endswith('.xlsx'):
    return df.to_excel(filename, **kwargs)
  elif filename.endswith('.csv'):
    return de.to_csv(filename, **kwargs)
  raise ValueError('File must be CSV or Excel')

def process(df, 
             max_length, 
             tokernizer, 
             label_name='label', 
             text_name='sentence',
             method):
  
  """
  Function to process text for pytorch
  """

  # Getting list of sentences and their labels

  if method == 'train':
    sentences = df[text_name].values
    labels = df[label_name].values
  else:
    sentences = df[text_name].values

  # Tokenizer all of the sentences and map the tokens to their word IDs
  input_ids = []
  attention_masks = []
  token_type_ids = []

  for sent in sentences:
    encoded_dict = tokenizer.encode_plus(sent,
                                         max_length=max_length,
                                         truncation=True,
                                         pad_to_max_length=True,
                                         return_attention_masks=True,
                                         return_token_type_ids=True,
                                         return_tensors='pt')
    
    # Adding each to a list
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    token_type_ids.append(encoded_dict['token_type_ids'])

  
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  token_type_ids = torch.cat(token_type_ids, dim=0)
  
  if method == 'train':
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids,
                            attention_masks,
                            token_type_ids,
                            labels)
  else:
    dataset = TensorDataset(input_ids, 
                            attention_masks,
                            token_types_ids)
  
  return dataset

def get_device():
  """
  Can we use cuda?
  """

  if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available" % torch.cuda.device_count())
    print("Using: ", torch.cuda.get_device_name(0))

  else:
    print("No GPU available, using CPU")

  return device

def flat_accuracy(preds, labels):
  """
  Get accuracy
  """
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()

  return np.sum(preds_flat == labels_flat) / len(labels_flat)

def do_prediction(model, dataloader, device):
  """
  Make predictions
  """

  preds, true_labels = [], []

  model.eval()
  for batch in dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch
    # Tell model not to compute or store gradients
    # Saving memory and speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids,
                      token_type_ids=b_token_type_ids,
                      attention_mask=b_input_mask)
    logits = outputs[0]
    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to("cpu").numpy()

    # Store predictions and true labels
    preds.append(logits)
    true_labels.append(label_ids)

  # Combine the results across batches
  flat_preds = np.concatenate(preds, axis=0)

  # For each sample, pick the label with the higher score
  flat_preds = np.argmax(flat_preds, axis=1).flatten()

  # Combine the correct labels for each batch into a single list
  flat_true_labels = np.concatenate(true_labels, axis=0)

  return pd.DataFrame({'predictions': flat_preds.tolist(),
                       'labels': flat_true_labels.tolist()})
  
def do_random_splitting(df,
                        val_size,
                        test_size=None,
                        label_name='label',
                        seed=None):
  
  """
  Split data into train, validation, and test (optional)
  """

  # default test set to None
  test_set, test_index = None, None

  # Test and validation size
  if test_size is not None:
    test_valid_size = test_size + val_size
    val_size = val_size / test_valid_size
  else:
    test_valid_size = val_size

  # Split data into train and val
  split = StratifiedShuffleSplit(n_splits=1,
                                 test_size=test_salid_size,
                                 random_state=seed)
  
  for train_index, test_valid_index in split.split(df, df[label_name]):
    train_set = df.iloc[train_index].copy()
    valid_set = df.iloc[test_valid_index].copy()

    train_set['split'] = 'train'
    valid_set['split'] = 'valid'

  splits = pd.concat((train_Set, valid_set))
  indexes = train_index.tolist() + test_valid_index.tolist()

  # If test, get test set
  if test_size is not None:
    split_test = StratifiedShuffleSplit(n_split=1,
                                        test_size=val_size,
                                        random_state=seed)
    for test_index, valid_index in split_test.split(valid_set,
                                                    valid_set[label_name]):
      test_set = valid_set.iloc[test_index].copy()
      valid_set = valid_set.iloc[valid_index].copy()

      test_set['split'] = 'test'
      valid_set['split'] = 'valid'

    splits = pd.concat((train_set, valid_set, test_set))['split'].tolist()
    indexes = train_index.tolist() + \
      test_index.tolist() + \
      valid_index.tolist()

  return train_set, valid_set, test_set, splits, indexes

def main()

  parser = argparse.ArgumentParser(description='Classification')

  # Arguments
  parser.add_argument('input_file', help='path to input')
  parser.add_argument('output_file', help='path to output')
  parser.add_argument('-mo', '--mode', default='train',
                      choices=['train', 'predict'],
                      help='Training of prediction')
  parser.add_argument('-e', '--epochs', default=4, type=int, help='Epochs')
  parser.add_argument('-b', '--batch', default=32, type=int, help='Batch size')
  parser.add_argument('-l', '--learning_rate', default=5e-5, type=float,
                      help='Learning rate. Options= 5e-5, 3e-5, 2e-5')
  parser.add_argument('-eps', '--epsilon', default=1e-8, type=float,
                      help='Epsilon')
  parser.add_argument('-w', '--warm_up', default=0.01, type=float,
                      help='Warmup % as a function of total steps')
  parser.add_argument('-m', '--max_sequence_length', default=128, type=int,
                      help='Max sequence length')
  parser.add_argument('-v', '--val_sze', default=0.1, type=float,
                      help='Validation size')
  parser.add_argument('-t', '--test_size', default=0.2, type=int,
                      help='Test size')
  parser.add_argument('-s', '--seed', default=42, type=int, 
                      default='Random Seed')
  
  args = parser.parse_args()

  # Output directory
  output_dir = os.path.dirname(os.path.abspath(args.output_file))

  # Get device
  device = get_device()
  print('----------------')
  print('Reading Data')
  print('----------------')
  df = read_file(args.input_file)


  #### TRACY: 
  #### Instead of print out everything to console, it is better if you log them to file
  #### It is easier for you to monitor the run on remote server,
  #### Or later when you want to look at the old experiments again

  # If we're predicting
  if args.mode == 'predict':

      print('----------------')
      print('Predicting')
      print('----------------')

    # Load model and tokenizer
    model = AutoModel.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Model to GPU
    model.to(device)

    # Process data
    pred_data = process(df, args.max_sequence_length, tokenizer, 'test')

    # Dataloader
    pred_dataloader = DataLoader(pred_data,
                                 sampler=SequentialSampler(pred_data),
                                 batch_size=args.batch)
    
    # Do predictions
    df_preds = do_prediction(model, pred_dataloader, device)

    # Create output directory if needed and write file
    os.makedir(output_dir, exists_ok=True)
    write_file(df_preds, args.output_file)

    # Exit program
    sys.exit()

  #### TRACY: 
  #### Are you missing "else" here? 
  #### In your code, even with "predict" mode, the model will be pretrained 

    # Training Mode
  print('----------------')
  print('Train-Validation-Test Split')
  print('----------------')

  (train_set,
   valid_set,
   test_set,
   splits,
   indexes) = do_random_splitting(df,
                                  args.val_size,
                                  args.test_size,
                                  seed=args.seed)
   
  print('{} training samples.'.format(train_set.shape(0)))
  print('{} validation samples.'.format(valid_set.shape(0)))
  print('{} testing samples.'.format(0 if test_set is None
                                     else test_set.shape(0)))
  
  print('----------------')
  print('Processing Data')
  print('----------------')

  # Initialize the data loader as a partial function
  load = partial(DataLoader, batch_size=args.batch)

  # Initialize the tokenizer

  #### TRACY: 
  #### You can have one more argument such as --model_name
  #### So when you want to do experiment with other models, you don't have to update the code
  tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

  # Process data
  train_data = process(train_set, args.max_sequence_length, tokenizer, 'train')
  val_data   = process(valid_set, args.max_sequence_length, tokenizer, 'train')

  if test_set is not None:
    test_data = process(test_set, args.max_sequence_length, tokenizer, 'test')

  # Create dataloader for our training and validation sets
  # We'll take training samples in random order
  train_dataloader = load(train_data, sampler=RandomSampler(train_data))

  # Validation set order does not matter. We'll read it sequentially
  val_dataloader = load(val_data, sampler=SequentialSampler(val_data))

  print('----------------')
  print('Data Processed')
  print('----------------')

  print('----------------')
  print('Building Model')
  print('----------------')

  # Initialize Model
  model = AutoModel.from_pretrained("distilroberta-base")

  # Tell pytorch to run this model on the GPU
  if torch.cuda.is_available():
    model.cuda()

  # Initialize the optimizer
  optimizer = AdamW(model.parameters(),
                    lr=args.learning_rate,
                    eps=args.epsilon)
  
  total_steps = len(train_dataloader) * args.epochs

  # We're using some warmup steps to help the model learn easier. 
  # According to: https://github.com/google-research/bert/issues/649
  # some people have been playing around with a value ~1%
  warmup_steps = total_steps * args.warmup
  print("{} Warmup steps".format(warmup_steps))

  # Initialize scheduler
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=warmup_steps,
                                              num_training_steps=total_steps)
  
  #### TRACY:
  #### Better to set seed in the beginning of the script
  #### Random seed also affects `train-val-test split`
  #### If you want to re-do the experiment with the same seed

  # Set random seed
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  training_stats = []

  total_t0 = time.time()

  for epoch_i in range(0, args.epochs):
    print('                 Training'              )
    print("=========== Epoch {:} / {:} ==========="
          ''.format(epoch_i + 1, args.epochs))
    t0 = time.time()
    total_train_loss = 0


    #### TRACY: 
    #### Is this `model.train()`?

    model.train()

    for step, batch in enumerate(train_dataloader):
      if step % 40 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print(" Batch {} of {}.     Elapsed: {:}"
              ''.format(step, len(train_dataloader), elapsed))
        
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_token_type_ids = batch[2].to(device)
      b_labels = batch[3].to(device)

      model.zero_grad()

      (loss, logits) = model(b_input_ids,
                           token_type_ids=b_token_type_ids,
                           attention_mask=b_input_mask,
                           labels=b_labels)
      
      total_train_loss += loss.item()

      # Perform backward pass to calculate gradients
      loss.backward()

      # Clip the norm of the gradients to 1.0
      # This is to help prevent exploding gradients
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      # Update the parameters and take a step using the computed gradient
      optimizer.step()

      # Update the learning rate
      scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    print(" Average training loss: {0:.2f}".format(avg_train_loss))
    print(" Training epoch took: {:}".format(training_time))

    # Validation
    print("Validation")
    t0 = time.time()

    total_eval_accuracy = 0
    total_eval_loss = 0

    model.eval()
    for batch in val_dataloader:
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_token_type_ids = batch[2].to(device)
      b_labels = batch[3].to(device)

      with torch.no_grad():
        (loss, logits) = model(b_input_ids,
                               token_type_ids=b_token_type_ids,
                               attention_mask=b_input_mask,
                               labels=b_labels)
        
      total_eval_loss += loss.item()

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to("cpu").numpy()

      #### TRACY:
      #### It is ok, but normally I also print out the Precision, Recall, F1 (sklearn.metrics.classification_report)
      #### for better understanding of the model result

      total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print(" Accuracy: {0:.2f}"..format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(val_dataloader)

    validation_time = format_time(time.time() - t0)

    print(" Validation Loss: {0:.2f}".format(avg_val_loss))
    print(" Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            "Erpoch": epoch_i + 1,
            "Training Loss": avg_training_loss,
            "Validation Loss": avg_val_loss,
            "Validation Accuracy": avg_val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time
          
        }
    )
  
  print("Training complete!")
  print("Total training took {:} (h:mm:ss)"
        "".format(format_time(time.time() - total_t0)))
  
  # Get a sequential dataloader for the training set
  train_dataloader = load(train_data, sampler=SequentialSampler(train_data))

  # Predict on the training and validation sets
  train_preds = do_prediction(model, train_dataloader, device)
  val_preds   = do_prediction(model, val_dataloader, device)

  # If there is a test set, then we use that, otherwise concatenate training
  # and validation samples
  if test_set is not None:
    test_dataloader = load(test_data, sampler=SequentialSampler(test_data))
    test_preds = do_prediction(model, test_dataloader, device)
    df_final = pd.concat((train_preds, test_preds, val_preds))
  else:
    df_final = pd.concat((train_preds, val_preds))

  # Add the splits and original indexes back into the data
  df_final["splits"] = splits
  df_final["indexes"] = indexes

  print("Done")
  print("Saving Model")

  # Create output directory if needed
  os.makedirs(output_dir, exist_ok=True)

  print(f"Saving model to {output_dir}")

  # Save model, configuration and tokenizer using 'save_pretrained()'
  model_to_save = model.module if hasattr(model, "module") else model
  model_to_save.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

  # Saving training arguments with trained model
  torch.save(args, os.path.join(output_dir, "training_args.bin"))

  # Write out the final file
  write_file(df_final, args.output_file)

if __name__ == '__main__':
  main()