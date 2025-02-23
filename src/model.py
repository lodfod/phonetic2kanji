from transformers import EncoderDecoderModel, BertTokenizer

# Load a pre-trained model as a starting point
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-multilingual-cased', 
                                                              'bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
