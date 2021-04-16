from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, CharacterEmbeddings, PooledFlairEmbeddings
#from MEAN_Bert import MEANBertEmbeddings
from typing import List
import torch
torch.cuda.empty_cache()
from flair.data import Corpus
from flair.datasets import ColumnCorpus


from flair.embeddings import BertEmbeddings
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import RoBERTaEmbeddings

embedding_types: List[TokenEmbeddings] = [
    #OpenAIGPT2Embeddings(),
    BertEmbeddings(layers = "-1,-2,-3,-4", pooling_operation = "mean"),
    #MEANBertEmbeddings(layers = "-1,-2,-3,-4",pooling_operation = "mean"),
    #RoBERTaEmbeddings(),
    #WordEmbeddings('glove'),
    #ELMoEmbeddings(),
    #CharacterEmbeddings(),
    #FlairEmbeddings('mix-forward'),
    #FlairEmbeddings('mix-backward'),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
#from flair.models import SequenceTagger
#tagger: SequenceTagger = SequenceTagger(hidden_size=1024, #1024
#                                        embeddings=embeddings,
#                                        tag_dictionary=tag_dictionary,
#                                        tag_type=tag_type,
#                                       use_crf=True)

# 6. initialize trainer
#from flair.trainers import ModelTrainer
#trainer: ModelTrainer = ModelTrainer(tagger, corpus)
print('bert works')

embedding_types: List[TokenEmbeddings] = [
    #OpenAIGPT2Embeddings(),
    #BertEmbeddings(layers = "-1,-2,-3,-4", pooling_operation = "mean"),
    #MEANBertEmbeddings(layers = "-1,-2,-3,-4",pooling_operation = "mean"),
    RoBERTaEmbeddings(),
    #WordEmbeddings('glove'),
    #ELMoEmbeddings(),
    #CharacterEmbeddings(),
    #FlairEmbeddings('mix-forward'),
    #FlairEmbeddings('mix-backward'),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
#from flair.models import SequenceTagger
#tagger: SequenceTagger = SequenceTagger(hidden_size=1024, #1024
##                                        embeddings=embeddings,
#                                        tag_dictionary=tag_dictionary,
#                                        tag_type=tag_type,
#                                        use_crf=True)

# 6. initialize trainer
#from flair.trainers import ModelTrainer
#trainer: ModelTrainer = ModelTrainer(tagger, corpus)
print('roberta works')

embedding_types: List[TokenEmbeddings] = [
    #OpenAIGPT2Embeddings(),
    #BertEmbeddings(layers = "-1,-2,-3,-4", pooling_operation = "mean"),
    #MEANBertEmbeddings(layers = "-1,-2,-3,-4",pooling_operation = "mean"),
    #RoBERTaEmbeddings(),
    WordEmbeddings('glove'),
    #ELMoEmbeddings(),
    #CharacterEmbeddings(),
    #FlairEmbeddings('mix-forward'),
    #FlairEmbeddings('mix-backward'),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
#from flair.models import SequenceTagger
#tagger: SequenceTagger = SequenceTagger(hidden_size=1024, #1024
#                                        embeddings=embeddings,
#                                        tag_dictionary=tag_dictionary,
#                                        tag_type=tag_type,
#                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer
#trainer: ModelTrainer = ModelTrainer(tagger, corpus)
print('glove works')

embedding_types: List[TokenEmbeddings] = [
    #OpenAIGPT2Embeddings(),
    #BertEmbeddings(layers = "-1,-2,-3,-4", pooling_operation = "mean"),
    #MEANBertEmbeddings(layers = "-1,-2,-3,-4",pooling_operation = "mean"),
    #RoBERTaEmbeddings(),
    #WordEmbeddings('glove'),
    ELMoEmbeddings(),
    #CharacterEmbeddings(),
    #FlairEmbeddings('mix-forward'),
    #FlairEmbeddings('mix-backward'),
]
# 5. initialize sequence tagger
#from flair.models import SequenceTagger
#tagger: SequenceTagger = SequenceTagger(hidden_size=1024, #1024
#                                        embeddings=embeddings,
#                                        tag_dictionary=tag_dictionary,
#                                        tag_type=tag_type,
#                                        use_crf=True)

# 6. initialize trainer
#from flair.trainers import ModelTrainer
#trainer: ModelTrainer = ModelTrainer(tagger, corpus)
print('elmo works')

embedding_types: List[TokenEmbeddings] = [
    #OpenAIGPT2Embeddings(),
    #BertEmbeddings(layers = "-1,-2,-3,-4", pooling_operation = "mean"),
    #MEANBertEmbeddings(layers = "-1,-2,-3,-4",pooling_operation = "mean"),
    #RoBERTaEmbeddings(),
    #WordEmbeddings('glove'),
    #ELMoEmbeddings(),
    CharacterEmbeddings(),
    #FlairEmbeddings('mix-forward'),
    #FlairEmbeddings('mix-backward'),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
#from flair.models import SequenceTagger
#tagger: SequenceTagger = SequenceTagger(hidden_size=1024, #1024
##                                        embeddings=embeddings,
#                                        tag_dictionary=tag_dictionary,
#                                        tag_type=tag_type,
#                                        use_crf=True)

# 6. initialize trainer
#from flair.trainers import ModelTrainer
#trainer: ModelTrainer = ModelTrainer(tagger, corpus)
print('char works')

embedding_types: List[TokenEmbeddings] = [
    #OpenAIGPT2Embeddings(),
    #BertEmbeddings(layers = "-1,-2,-3,-4", pooling_operation = "mean"),
    #MEANBertEmbeddings(layers = "-1,-2,-3,-4",pooling_operation = "mean"),
    #RoBERTaEmbeddings(),
    #WordEmbeddings('glove'),
    #ELMoEmbeddings(),
    #CharacterEmbeddings(),
    FlairEmbeddings('mix-forward'),
    FlairEmbeddings('mix-backward'),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
# 5. initialize sequence tagger
#from flair.models import SequenceTagger
#tagger: SequenceTagger = SequenceTagger(hidden_size=1024, #1024
#                                        embeddings=embeddings,
#                                        tag_dictionary=tag_dictionary,
#                                        tag_type=tag_type,
#                                        use_crf=True)

# 6. initialize trainer
#from flair.trainers import ModelTrainer
#trainer: ModelTrainer = ModelTrainer(tagger, corpus)
print('flair works')