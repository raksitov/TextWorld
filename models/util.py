import numpy as np


def words_to_ids(words, word_ids):
  ids = []
  for word in words:
    try:
      ids.append(word_ids[word])
    except KeyError:
      ids.append(word_ids['<UNK>'])
  return ids


def preproc(s, str_type='None', tokenizer=None, lower_case=True):
  if s is None:
    return ['nothing']
  s = s.replace('\n', ' ')
  if s.strip() == '':
    return ['nothing']
  if str_type == 'feedback':
    if '$$$$$$$' in s:
      s = ''
      if '-=' in s:
        s = s.split('-=')[0]
  s = s.strip()
  if len(s) == 0:
    return ['nothing']
  tokens = [t.text for t in tokenizer(s)]
  if lower_case:
    tokens = [t.lower() for t in tokens]
  return tokens


def max_len(list_of_list):
  return max(map(len, list_of_list))


def pad_sequences(sequences, max_len=None, dtype='int32', value=0.):
  '''
  Partially borrowed from Keras
  # Arguments
      sequences: list of lists where each element is a sequence
      max_len: int, maximum length
      dtype: type to cast the resulting sequence.
      value: float, value to pad the sequences to the desired value.
  # Returns
      x: numpy array with dimensions (number_of_sequences, max_len)
  '''
  lengths = [len(s) for s in sequences]
  nb_samples = len(sequences)
  if max_len is None:
    max_len = np.max(lengths)
  # take the sample shape from the first non empty sequence
  # checking for consistency in the main loop below.
  sample_shape = tuple()
  for s in sequences:
    if len(s) > 0:
      sample_shape = np.asarray(s).shape[1:]
      break
  x = (np.ones((nb_samples, max_len) + sample_shape) * value).astype(dtype)
  for idx, s in enumerate(sequences):
    if len(s) == 0:
      continue  # empty list was found
    # pre truncating
    trunc = s[-max_len:]
    # check `trunc` has expected shape
    trunc = np.asarray(trunc, dtype=dtype)
    if trunc.shape[1:] != sample_shape:
      raise ValueError(('Shape of sample %s of sequence at position %s is different' +
                        'from expected shape %s').format(trunc.shape[1:], idx, sample_shape))
    # post padding
    x[idx, :len(trunc)] = trunc
  return x


def get_embeddings(config):
  with open(config['glove_path']) as glove:
    glove_vocab = {}
    for vector in glove:
      vector = vector.lstrip().rstrip().split(' ')
      glove_vocab[vector[0]] = list(map(float, vector[1:]))
  with open(config['vocab_path']) as vocab:
    word_vocab = vocab.read().split('\n')

  embedding_matrix = np.zeros((len(word_vocab), config['glove_dim']))
  word_ids = {}
  for idx, word in enumerate(word_vocab):
    word_ids[word] = idx
    if word not in glove_vocab:
      if idx:
        embedding_matrix[idx, :] = np.random.randn(config['glove_dim'])
    else:
      embedding_matrix[idx, :] = glove_vocab[word]

  return embedding_matrix, word_ids


def dict_to_array(d, size):
  return [{k: v[idx] for (k, v) in d.items()} for idx in range(size)]
