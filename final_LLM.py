# Install PyTorch from your terminal with:
# python -m pip install torch torchvision torchaudio
import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm

try:
    import torch_directml
except ImportError:
    torch_directml = None

batch_size=64
block_size=256
max_iters=5000
eval_interval=500
learning_rate=3e-4

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU")
elif torch_directml is not None:
    device = torch_directml.device()
    print("Using AMD GPU via DirectML")
else:
    device = torch.device("cpu")
    print("Using CPU")

eval_iters=200
n_embd=384
n_head=6
n_layer=6
dropout=0.2
torch.manual_seed(1337)

#opens data file
with open("input.txt", "r") as file:
    text = file.read()

#applying sentence piece tokenizer to the text
# 1. Train a model directly from a raw text file.
# (No pre-tokenization or language-specific preprocessing required!)
spm.SentencePieceTrainer.train(
    input='input.txt', 
    model_prefix='m', 
    vocab_size=1000
)

# 2. Load the trained model.
sp = spm.SentencePieceProcessor(model_file='m.model')

# 3. Encode raw text into subword pieces (strings) or vocabulary IDs (integers).
# The model needs integer IDs, not strings.
pieces = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)
vocab_size = sp.vocab_size()
    
"""
#extract all unique characters that occur in text
chars=sorted(list(set(text)))
vocab_size=len(chars)
print(''.join(chars))
print(vocab_size)

#mapping from characters to integers
stoi={ch:i for i, ch in enumerate(chars)}
itos={i:ch for i, ch in enumerate(chars)}
encode=lambda s: [stoi[c] for c in s]
decode=lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
#the most simplest tokenizer--> should look into tokenizer used by google
"""

#splitting data into train and test using integer token IDs
n=int(0.9*len(ids))
train_data=torch.tensor(ids[:n], dtype=torch.long)
val_data=torch.tensor(ids[n:], dtype=torch.long)

#extracting batches from training data (data loading)
def get_batch(split):
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size, (batch_size,))#torch.randint(range,numberofrandomintegers)-->range= 0 to range
    x=torch.stack([data[i:i+block_size]for i in ix])
    y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits, loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out
"""cons of bigram model:
It has no long-term memory. Because it only looks one step into the past, 
generated text quickly loses track of grammar, structure, and meaning, often resulting in repetitive or nonsensical gibberesqe."""

class Head(nn.Module):
    #one head of self attention

    def __init__(self, head_size):
        super().__init__()
        self.key=nn.Linear(n_embd, head_size, bias=False)
        self.query=nn.Linear(n_embd, head_size, bias=False)
        self.value=nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        B,T,C=x.shape #C is head size
        k=self.key(x) #(B,T,C)
        q=self.query(x) #(B,T,C)
        v=self.value(x) #(B,T,C)
        #compute attention scores ("affinities")
        wei=q @ k.transpose(-2,-1) * C** -0.5 #(B,T,C) @ (B,C,T)=(B,T,T)
        wei=wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei=F.softmax(wei, dim=-1) # (B,T,T)
        wei=self.dropout(wei)
        #perform the weighted aggregation of the values
        out=wei @ v #(B,T,T) @ (B,T,C)=(B,T,C)
        return out
    
class MultiHeadAttention(nn.Module):
    #multiple heads of self attention
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(n_embd, n_embd)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        out=torch.cat([h(x) for h in self.heads], dim=-1)
        out=self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size=n_embd//n_head
        self.sa_heads=MultiHeadAttention(n_head,head_size)# i.e. n heads of head_size dimensional self attention
        self.ffwd=FeedForward(n_embd)
        self.layer_norm1=nn.LayerNorm(n_embd)
        self.layer_norm2=nn.LayerNorm(n_embd)

    def forward(self, x):
        x=x+self.sa_heads(self.layer_norm1(x))#apply one head of self_attention (B,T,C)
        x=x+self.ffwd(self.layer_norm2(x)) #B,T,C
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table=nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table=nn.Embedding(block_size, n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        
        self.lm_head=nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T=idx.shape
        tok_embd=self.token_embedding_table(idx)#(B,T,C) here C is n_embd
        pos_embd=self.position_embedding_table(torch.arange(T, device=device))#T,C
        x=tok_embd+pos_embd #(B,T,C)
        x=self.blocks(x)#B,T,C
        logits=self.lm_head(x) #(B,T, vocab_Size)
        
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits, targets)
        return logits,loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #get the predictions
            idx_cond=idx[:,-block_size:] #crop idx to the last block_size tokens
            logits, loss= self(idx_cond)
            #focus only on last time step
            logits=logits[:,-1,:]#becomes B,C
            #apply softmax to get probabilities
            probs=F.softmax(logits, dim=1)#B,C
            #sample from the distribution
            idx_next=torch.multinomial(probs, num_samples=1)#B,1

            """If you always picked the highest score using something like torch.argmax(), 
            your language model would become completely deterministic. By using torch.multinomial,
            you introduce creativity and randomness. Even if one character is the favorite, 
            the model still has a small chance to pick an alternative, 
            making the generated text feel much more natural and varied!"""

            #append sampled index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1)
        return idx


model=BigramLanguageModel()
m=model.to(device)

#optimization (minimization of loss)
optimizer=torch.optim.AdamW(m.parameters(),lr=learning_rate)
#highly advanced, smart optimizer as compared to SGD which is looks at only the current gradient and has no memory of past steps
for iter in range(max_iters):

    #every once in a while evaluate the loss on train and val sets
    if iter%eval_interval==0:
        losses=estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    #sample a batch of data
    xb, yb=get_batch('train')

    #evaluate the loss
    logits, loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

#model after optimization
context=torch.zeros((1,1),dtype=torch.long, device=device)
print(sp.decode(m.generate(context, max_new_tokens=3000)[0].tolist() ))