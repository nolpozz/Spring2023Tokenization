from tokenizers import Tokenizer
from tokenizers.models import BPE


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))


from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])




from tokenizers.pre_tokenizers import BertPreTokenizer
tokenizer.pre_tokenizer = BertPreTokenizer()

#files = [f"HFUwgetStuff/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
files = ["HFUwgetStuff/wikitext-103-raw/wiki.test.raw","HFUwgetStuff/wikitext-103-raw/wiki.train.raw","HFUwgetStuff/wikitext-103-raw/wiki.valid.raw"]
tokenizer.train(files, trainer)


tokenizer.save("tokenizer-wiki.json")

from tokenizers.trainers import UnigramTrainer



output = tokenizer.encode("Hello, y'all! How are you 😁 ?")


print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]

print(output.ids)
# [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]

print(output.offsets[9])
# (26, 27)


sentence = "Hello, y'all! How are you 😁 ?"
sentence[26:27]
# "😁"



tokenizer.token_to_id("[SEP]")
# 2


from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)