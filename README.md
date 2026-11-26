# Sample University Projects

* `a3-BiLSTM.py`:  implementation of a bidirectional LSTM network for the task of segmenting compound words in German

* `a4-transformer`: training an encoder-decoder transformer model using OpenNMT for the task of phoneme conversion (G2P)
  * Only the core files were included, for simplicity
  * `data_processing.py`: preprocessing of the training and test sets, reserving part of the training data for validation
  * `a4-transformer.yaml`: configuration file for OpenNMT
  * `a4-transformer.py`: running OpenNMT programmatically; converting predictions into expected format
  * `a4.test`: the test set, contains one word (sequence of graphemes) per line
  * `a4-transformer.predictions`: the predictions file, contains the original word and its predicted IPA transcription
  
* `projective_dependencies.py`: implementation of various methods working with dependency trees and (non) projectivity in CoNLL-U format
