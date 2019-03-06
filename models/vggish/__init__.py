# from __future__ import print_function
# import numpy as np
# import librosa
#
# import .vggish_input
# import .vggish_params
# import .vggish_postprocess
# import .vggish_keras

# # Paths to downloaded VGGish files.
# checkpoint_path = 'vggish_weights.ckpt'
# pca_params_path = 'vggish_pca_params.npz'
#
# # Produce a batch of log mel spectrogram examples.
# wav_fn = 'example.wav'
# x, sr = librosa.load(wav_fn)
# input_batch = vggish_input.waveform_to_examples(x, sr)
# print('Log Mel Spectrogram example: ', input_batch[0])
#
# # Define VGGish, load the checkpoint, and run the batch through the model to
# # produce embeddings.
# # model = vggish_keras.vggish_keras_sequential()
# model = vggish_keras.get_vggish_keras()
# model.load_weights(checkpoint_path)
# embedding_batch = model.predict(input_batch[:,:,:,None])
# print('VGGish embedding: ', embedding_batch[0])
#
# # Postprocess the results to produce whitened quantized embeddings.
# pproc = vggish_postprocess.Postprocessor(pca_params_path)
# postprocessed_batch = pproc.postprocess(embedding_batch)
# print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
#
# print('\nLooks Good To Me!\n')
