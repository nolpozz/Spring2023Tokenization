

from tokenizers import normalizers 
from tokenizers.normalizers import BertNormalizer

normalizer = normalizers.BertNormalizer
##print(normalizer.normalize_str("Héllò hôw are ü?"))




from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Digits
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation, Digits(individual_digits=True)])





from tokenizers import Tokenizer
from tokenizers.models import Unigram
tokenizer = Tokenizer(Unigram)

tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer


from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)


from tokenizers import Tokenizer
##from tokenizers.models import Unigram
bert_tokenizer = Tokenizer(Unigram(unk_token="[UNK]"))


from tokenizers import normalizers
from tokenizers.normalizers import BertNormalizer
bert_tokenizer.normalizer = normalizers.BertNormalizer

from tokenizers.pre_tokenizers import Whitespace, Punctuation, Digits
bert_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation, Digits(individual_digits=True)])

from tokenizers.processors import TemplateProcessing
bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)



from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
##files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
files = ["HFUwgetStuff/wikitext-103-raw/wiki.test.raw","HFUwgetStuff/wikitext-103-raw/wiki.train.raw","HFUwgetStuff/wikitext-103-raw/wiki.valid.raw"]
bert_tokenizer.train(files, trainer)
bert_tokenizer.save("bert-wiki.json")



output = tokenizer.encode("Hello, y'all! ы How are you?")
print(output.ids)
# [1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2]
tokenizer.decode([1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2])
# "Hello , y ' all ! How are you ?"



output = bert_tokenizer.encode("Welcome to the Tokenizers library ы .")
print(output.tokens)
# ["[CLS]", "welcome", "to", "the", "[UNK]", "tok", "##eni", "##zer", "##s", "library", ".", "[SEP]"]
bert_tokenizer.decode(output.ids)
# "welcome to the tok ##eni ##zer ##s library ."


from tokenizers import decoders
bert_tokenizer.decoder = decoders.WordPiece()
bert_tokenizer.decode(output.ids)
# "welcome to the tokenizers library."