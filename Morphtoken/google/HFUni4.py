from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
tokenizer = Tokenizer(models.Unigram())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tokenizer.decoder = decoders.Metaspace()
trainer = trainers.UnigramTrainer(
    vocab_size=20000,
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
)



# from datasets import load_dataset_builder
# ds_builder = load_dataset_builder("rotten_tomatoes")

# ds_builder.info.description

# ds_builder.info.features

# from datasets import load_dataset

# dataset = load_dataset("rotten_tomatoes")


















# from urllib.request import urlretrieve

# url = "http://archivo.dbpedia.org/download?o={ontologyURI}"
# urlretrieve(url, 'file.txt')

# tokenizer.train('file.txt')


# import bz2

# bz_file = bz2.BZ2File("small.txt.bz2")
# lines = bz_file.readlines()
# for line in lines:
#     line = line.rstrip('\n')
#     label, text = line.split('\t')
#     text_words = text.split(',')
#     print(label)

# with bz2.open("newsSpace.bz2", "rt") as bz_file:
#     tokenizer.train(bz_file)



import datasets
dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train+test+validation")

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))







# import gzip
# # with gzip.open("data/my-file.0.gz", "rt") as f:
# #     tokenizer.train_from_iterator(f, trainer=trainer)

# files = ["sentencepiece/data/my-file.0.gz", "sentencepice/data/my-file.1.gz", "sentencepiece/data/my-file.2.gz"]
# def gzip_iterator():
#     for path in files:
#         with gzip.open(path, "rt") as f:
#             for line in f:
#                 yield line
# tokenizer.train_from_iterator(gzip_iterator(), trainer=trainer)