from metaflow import FlowSpec, step
from baseline_model import forward, Transformer

class LinearFlow(FlowSpec):

    @step
    def start(self):
        self.my_var = 'hello world'
        self.next(self.preprocess)

    @step
    def preprocess(self):
        self.foo = open("TinyStories-valid.txt", "r")
        self.content = self.foo.readlines()
        self.con_str = ''
        for c in self.content:
          self.con_str += c
        self.stories = self.con_str.split("<|endoftext|>\n")
        self.text = "\n".join(self.stories)
        self.next(self.vocabulary)

    @step
    def vocabulary(self):
        import torch
        a = torch.float32
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i, ch in enumerate(self.chars) }
        self.itos = {i:ch for i,ch in enumerate(self.chars)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: "".join([self.itos[x] for x in l])
        self.data = torch.tensor(self.encode(self.text), dtype = torch.long)
        self.next(self.data_preparation)

    @step
    def data_preparation(self):
        import torch

        self.n = int(0.9*len(self.data))
        self.train_data = self.data[:self.n]
        self.val_data = self.data[self.n:]
        self.block_size = 8
        self.x = self.train_data[:self.block_size]
        self.y = self.train_data[1:self.block_size+1]
        for t in range(self.block_size):
            context = self.x[:t+1]
            target = self.y[t]
        torch.manual_seed(1337)
        self.batch_size = 4
        self.next(self.loading_data)

    @step
    def get_training_batch(self):
        import torch

        self.ix = self.torch.randint(len(self.train_data) - self.block_size, (self.batch_size,))
        self.xb = torch.stack([self.train_data[i:i+self.block_size] for i in self.ix])
        self.yb = torch.stack([self.train_data[i+1:i+self.block_size+1] for i in self.ix])
        self.device= 'cuda'
        for b in range(self.batch_size):
            for t in range(self.block_size):
                self.context = self.xb[b][:t+1]
                self.target = self.yb[b][t]
        self.next(self.baseline_moe)

    
    @step
    def baseline_moe(self):
        self.next(self.training_hyperparameters)

    @step 
    def training(self):
        import torch

        self.batch_size = 64 # how many independent sequences will we process in parallel?
        self.block_size = 256 # what is the maximum context length for predictions?
        self.max_iters = 5000
        self.eval_interval = 100
        self.learning_rate = 1e-3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = 200
        self.n_embd = 384
        self.n_embed = 384
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.0
        self.low_rank = True
        self.model = Transformer()
        self.lora.mark_only_lora_as_trainable(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.learning_rate)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=1e-4)

        for iter in range(self.max_iters):
            if iter % 100 == 0 or iter == self.max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        self.xb = self.xb.to(self.device)
        self.yb = self.yb.to(self.device)

        # evaluate the loss
        self.logits, self.loss = self.model(self.xb, self.yb)
        self.optimizer.zero_grad(set_to_none=True)
        self.loss.backward()
        self.optimizer.step()
        self.next(self.testing_model)



    @step
    def testing_model(self):
        import torch

        self.d = 'once upon a time there was a '
        self.x = torch.tensor(self.encode(d), dtype = torch.long,device=self.device).unsqueeze(0)
        self.print(self.decode(self.model.generate(self.x, max_new_tokes=500)[0].tolist()))
        self.next(self.end)

    @step
    def end(self):
        print('Model Trained using metaflow')

if __name__ == '__main__':
    LinearFlow()