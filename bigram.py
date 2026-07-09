# Install PyTorch from your terminal with:
# python -m pip install torch torchvision torchaudio
import torch
import torch.nn as nn
from torch.nn import functional as F

#parameters
batch_size=32
block_size=8
max_iters=3000
eval_interval=300
learning_rate=1e-3
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
torch.manual_seed(1337)

#opens data file
with open("input.txt", "r") as file:
    text = file.read()

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

#splitting data into train and test
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

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
"""cons of biagram model:
It has no long-term memory. Because it only looks one step into the past, 
generated text quickly loses track of grammar, structure, and meaning, often resulting in repetitive or nonsensical gibberesqe."""

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits=self.token_embedding_table(idx)
        
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
            logits, loss= self(idx)
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
context=idx=torch.zeros((1,1),dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist() ))