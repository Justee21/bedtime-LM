# Bedtime-LM

Justin Roberts ([Justee21](https://github.com/Justee21)) and I ([ericyenn](https://github.com/ericyenn)) created this to better understand how transformers work internally and to also learn PyTorch. We learned the content from class, *Attention Is All You Need*, and our preferred agents. We handwrote all the code and left our comments in the code. This README was created via the notes we wrote in the code + the shared doc that we had while building this. Spent about a week.

## Current Parameters

| | |
|---|---|
| Tokenizer | Byte-level BPE |
| Embedding dimension | $384$ |
| Attention heads | $6$ |
| Transformer blocks | $2$ MoE followed by $4$ dense |
| Experts per MoE block | $8$, with $2$ selected per token |
| Context length | $128$ tokens |

## Notebook Structure

Everything lives in `optimusPrime.ipynb` (built to run on Google Colab with Google Drive mounted).

1. **Setup** – imports, choosing the device, mounting drive, and setting the checkpoint path
2. **Load tokenizer and data** – uses a saved tokenizer and memory-mapped training and validation files (these files are from the TinyStories dataset uploaded from HF). Earlier data-preparation experiments are retained in the appendix.
3. **Construct training examples** – create input sequences and next-token targets from the token files.
4. **Define the architecture** – causal multi-head attention with rotary positional embeddings, the dense transformer block, the MoE transformer block, and the complete model.
5. **Define the training and validation** – defines `validation_loop` before `train_loop` since the validation is tested throughout the training loop.
6. **Configuring the experiment** – constructs the data loaders, model, loss function, and optimizer. Running this section initializes a new model and optimizer.
7. **Train and save** – trains a new model and saves its checkpoint. To use an existing checkpoint, skip this training cell and continue to the checkpoint-loading section.
8. **Load a checkpoint** – loads the checkpoint at the configured path for evaluation or generation. If you are continuing directly from training, you can proceed to evaluation.
9. **Evaluation** – evaluates the current model on the validation subset.
10. **Inference!** – defines the generation, then runs the example prompt using the current model and tokenizer.
11. **Appendix: other experiments and checkpoints** – these cells are separate from the main workflow and are just meant to be references to what's been done earlier. Run only the cells needed for a specific experiment; executing the entire appendix can overwrite the active tokenizer, data variables, or model state.

## Architecture Notes

### Dataset

We created a custom dataset so each user can determine the context size that they want to deal with. Suppose you have a tensor of size $N$ and context size of length $L$, then the highest index that should be allowed is $N-L$, and when you get an item it is the tensor spliced from the index to the context size, and the target is shifted one to the right.

### Multi-head attention

Inputting the embedding table $X$ with dimensions $(B, T, d_{model})$, we also want an output of the same size. We have learned weight matrices $W_Q, W_K, W_V$ and multiply $X$ by each to get $Q, K, V$.

The idea of the shaping: $X$ is of shape $(B, T, D)$ after the embeddings, and in each head the learned matrices $W_K, W_Q, W_V$ are of size $D \times d_k$, where $`d_k = D / \text{num\_heads}`$ (the head dim). So $XW_K$ is of size $(B, T, d_k)$, and then we concatenate all the heads and multiply by a $W_O$ matrix to get back to $(B, T, D)$.

The number of heads is usually 8 or 16 for smaller models; more modern GPTs are more in the 32–96 range.

### RoPE

Since RoPE requires us to compute the angles $\theta_k = 10000^{-2k/d}$, where $k$ represents the $k$-th $2 \times 2$ rotation block along the diagonal, we precompute all of the $k$ values ahead of time so we don't need to compute them in the forward pass (we need to do it `head_dim // 2` times).

In the forward pass we splice $Q$ and $K$ into even and odd dimensions and simplify the rotation matrix to an elementwise multiply:

```math
\begin{aligned}
q_{\text{even}}' &= q_{\text{even}} \cos\theta - q_{\text{odd}} \sin\theta \\
q_{\text{odd}}' &= q_{\text{even}} \sin\theta + q_{\text{odd}} \cos\theta
\end{aligned}
```

We sum over the hidden dim in $QK^\top$, so the order in which we keep the hidden dims doesn't matter.

### Transformer blocks

For each block, we pre-layer norm $X$ and then run MHA on it, then add $X$ back (residual). The FFN uses SwiGLU, and dropout ($0.2$) is applied right after attention and after the FFN.

The MoE blocks replace the single FFN with $8$ SwiGLU experts. A router picks the top $2$ experts per token, and their outputs are weighted by a softmax over the selected router scores. A load-balancing auxiliary loss is added to the cross-entropy loss with weight $0.01$.

### Complete model

We defined the embedding table, which in the beginning is completely random since these weights are learned and are what we will be optimizing. The `input_ids` represent the batch of sequences that we want to train over, where $B$ is the batch size and $T$ is the length of the sequence. For each token in our input, we fetch the corresponding embedding vector from the embedding table.

The final linear layer is tied to the embedding table, so we have the same weights that bring the token ids to the vector embedding as the weights that bring the vectors back to logits. The embedding table is initialized with standard deviation $0.02$.

The biggest challenge we had here was understanding the embedding table and understanding how to use it properly.

### KV cache

During inference, only the new token is processed after the first step, and its $K$ and $V$ are appended to the cache. Once the sequence gets longer than the context length, we remove the oldest entry to make room for the new token. RoPE positions keep increasing even when old cache entries are removed.

### Validation metrics

Besides the loss, we track next-token accuracy and whether one of the top 3 predicted tokens is correct. We have a $(B, T, 3)$ tensor of top-3 predictions and want to see if each $(b, t)$ vector has one of its values equal to the target, so we reshape the target into $(B, T, 1)$, compare, and count it as correct if any of the 3 match.

## Development Log

### Embedding dimension 48

```
validation_loop(validation_loader, model, loss_func, 64)
#for batchsize = 64, embed_dims = 48, seq_length=128, num_heads=4, num_layers=3
#total loss: 2.3619683620955656, accuracy of next token prediction: 0.3068228035355258%,
#accuracy that one of the top 3 predicted tokens is correct: 0.5473197877024019

total loss: 1.8135585044652938, accuracy of next token prediction: 0.467479308220838%,
 accuracy that one of the top 3 predicted tokens is correct: 0.6956751741284601
```

For the embedding dimension of 48, looking at every sequence, 3 epochs, and the Adam optimizer:

```python
prompt = "Shall I compare thee to a summer's day? Thou art more lovely and more temperate: Rough winds do shake the darling buds of May,"
inference_loop(prompt, model, 1, 128, 100)
```
```
Shall I compare thee to a summer's day? Thou art more lovely and more temperate: Rough winds do shake the darling buds of May,
Far as fate this we on'er modest;
Wolk, preck our you me, and be lapinish.
If That seame, Antell th
```

*Inference results*

We tried adding some sort of early stopping (where we check every 250 steps to see if there is a certain amount of decrease in the validation loss. If there isn't for 5 checks in a row, or if the loss increases, we stop training) since after training for a while, our model seemed to overfit on our data. However, this made the model even worse.

We then tried to randomly index the start positions of our training/validation data.

```
 accuracy that one of the top 3 predicted tokens is correct: 0.7400886194029851
170, 0, 1.3487242460250854
171, 0, 1.338688850402832
172, 0, 1.349787712097168
173, 0, 1.387675404548645
total loss: 1.6684049825033282, accuracy of next token prediction: 0.5252493541905855%,
 accuracy that one of the top 3 predicted tokens is correct: 0.7413353903559128
174, 0, 1.352958083152771
175, 0, 1.3584665060043335
176, 0, 1.347041130065918
177, 0, 1.350999116897583
total loss: 1.6622682802986468, accuracy of next token prediction: 0.5272047215843858%,
 accuracy that one of the top 3 predicted tokens is correct: 0.7429678530424799
1.659211793942238
```

Better than before, but still not good enough. We then made a few edits such as adding dropout right after the FFN and attention, changing hyperparameters to those of nanoGPT, and also changing the standard deviation to be 0.02 for the embedding table.

### SwiGLU, RoPE, and BPE

However, we were still having accuracy issues, so we also implemented SwiGLU.

Implemented RoPE cuz we learned it in class, and also switched to a new tokenizer that uses BPE rather than just character tokens. Here are some of the results we had from adding SwiGLU, RoPE, and BPE:

```
0, 2400, 2.928718090057373
total loss: 3.8541388154029845, accuracy of next token prediction: 0.2941162109375%,
 accuracy that one of the top 3 predicted tokens is correct: 0.4188720703125
0, 2500, 2.885547161102295
0, 2600, 2.758359432220459
0, 2700, 2.795011520385742
total loss: 3.898643672466278, accuracy of next token prediction: 0.295697021484375%,
 accuracy that one of the top 3 predicted tokens is correct: 0.42169189453125
0, 2800, 2.6944239139556885
0, 2900, 2.607179880142212
total loss: 3.9632219195365908, accuracy of next token prediction: 0.2936279296875%,
 accuracy that one of the top 3 predicted tokens is correct: 0.41790771484375
0, 3000, 2.634154796600342
0, 3100, 2.5489044189453125
0, 3200, 2.472245216369629
total loss: 4.031412839889526, accuracy of next token prediction: 0.29300537109375%,
```

Here are the hyperparameters we used:

```python
batch_size = 64
max_steps = 5000
context_size = 128

new_training_data = CharacterDataset(training_data, context_size)
new_validation_data = CharacterDataset(validation_data,context_size)

sampler = RandomSampler(new_training_data, replacement=True, num_samples = max_steps*batch_size)

training_loader = DataLoader(new_training_data, batch_size=batch_size, sampler=sampler)
#this is to act more similarly to nanogpt and choose randomly
#selected indices with replacement, idea here is choose batch_size * max_steps samples which here is 64*5000=320,000
#so we have that many sequences we check through


generator = torch.Generator().manual_seed(123) #this allows for more optimal
val_indices = torch.randint(high=len(new_validation_data), size=(20* batch_size,), device="cpu", generator=generator).tolist()
#this gives random but reproducible validation indices to test on

validation_loader = DataLoader(Subset(new_validation_data, val_indices), batch_size=batch_size, shuffle=False)
#this can just be used during training_loop and separate full validation test can be evaluated later

model = Model(vocab_size=vocab_size, embed_dims = 384, seq_length=128, MLP_dims=4*384, num_heads=6, num_layers=6).to(device) #hyperparameters
loss_func = nn.CrossEntropyLoss() #design  choice here
optimizer = torch.optim.AdamW(model.parameters(), lr=.0001) #design choice here
```

Here is the output:

```python
prompt = "Shall I compare thee to a summer's day? Thou art more lovely and more temperate: Rough winds do shake the darling buds of May,"
inference_loop(prompt, model, .7, 128, 1000)
```
```
Shall I compare thee to a summer's day? Thou art more lovely and more temperate: Rough winds do shake the darling buds of May,
And fourn the second stealful last of coward,
The mighty widow of my valour's grandses,
Unless you have read a poward in the field
And that the queen hath touch'd my powers of blood,
And that my brother loves me Tranio's gates.

QUEEN ELIZABETH:
Then sickly tell me, I am not a woman.

LADY ANNE:
Then am I all, my lord, for I am strange.

GLOUCESTER:
Why, that's a friend.

GLOUCESTER:
Softly, as I see, I'll have a single thing;
For I am glad to get a cloud of thy heart.

QUEEN ELIZABETH:
It is your grace.

GLOUCESTER:
His sovereign is not so much in this fiery father,
```

Even though the loss was way higher, our inference was much better (we did change the temperature to 0.7, but we did test on a temperature of 0.7 on previous iterations). We thought that this was mainly because of the tokenizer.

At this point, we also bought Colab+ because we ran out of GPU 😞

### TinyStories

We decided that our model was good enough for the Tiny Shakespeare dataset, so we decided to use a larger dataset: TinyStories.

Here is how we ended after 10k iterations:

```
validation_loop(validation_loader, model, loss_func, 64)
#for batchsize = 64, embed_dims = 48, seq_length=128, num_heads=4, num_layers=3
#total loss: 2.3619683620955656, accuracy of next token prediction: 0.3068228035355258%,
#accuracy that one of the top 3 predicted tokens is correct: 0.5473197877024019

total loss: 1.731829571723938, accuracy of next token prediction: 0.576727294921875%,
 accuracy that one of the top 3 predicted tokens is correct: 0.766888427734375
1.731829571723938
```

```
 accuracy that one of the top 3 predicted tokens is correct: 0.7638671875
0, 9500, 1.8511444330215454
0, 9600, 1.780876874923706
0, 9700, 1.8273837566375732
total loss: 1.7415041029453278, accuracy of next token prediction: 0.57557373046875%,
 accuracy that one of the top 3 predicted tokens is correct: 0.765167236328125
0, 9800, 1.7834854125976562
0, 9900, 1.809267520904541
total loss: 1.7374642491340637, accuracy of next token prediction: 0.576434326171875%,
 accuracy that one of the top 3 predicted tokens is correct: 0.766162109375
0, 10000, 1.8864672183990479
0, 10100, 1.790828824043274
0, 10200, 1.855654001235962
total loss: 1.7318295657634735, accuracy of next token prediction: 0.576727294921875%,
 accuracy that one of the top 3 predicted tokens is correct: 0.766888427734375
1.7318295657634735
```

Here is the inference output:

```python
prompt = "Once upon a time there was a story about Alice and Mark and how they had an encounter with a bear and they were very frightened "
inference_loop(prompt, model, .7, 128, 1000)
```
```
Once upon a time there was a story about Alice and Mark and how they had an encounter with a bear and they were very frightened — they were just trying to find their way out.
One day, Alice and Mark decided to go on an adventure. They walked and walked until they reached a big cave. Alice and Alice were very surprised. They did not know what was there.
Alice and Alice on the cave were excited to see who could help. The next day, Alice and Alice went for a walk in the woods. Alice was excited to have a new adventure.
<|endoftext|>
Once upon a time, in a small forest, there was a little boy named Tim. Tim loved to play in the forest. Every day, he would run and jump and jump, and Tim would sit on his back, and Sue would laugh.
One day, Tim saw a big, red ball in the forest. He wanted to play with it. He tried to throw the ball into the tree, but it was too late. The ball hit the ball and broke into pieces. Tim was sad and cried.
Tim's mom heard the noise and came to see what was wrong. She saw Tim and Sue fighting over the ball. She got out of the tree and said, "Tim, you should not play with your toys. They are not for playing with." Tim felt sad an...
<|endoftext|>
Once upon a time, there was a little girl named Lily. She loved to play with her toy box. One day, she found a big, pretty button. Lily was very happy and wanted to show her friend, Tom.
Lily went to Tom's house and said, "Look, Tom! I found a button with a star on it!" Tom looked at the star again and smiled. "That's a big, red button!" he said. Lily was happy to share the button with Tom. They knew that the...
Lily and Tom decided to put the button together. They put on their hat and a big smile. They were happy to have a new friend. They played with the magic button every day, and they had lots of fun.
<|endoftext|>
One day, a little boy named Tim went to the park with his mom. They played on the swings and slides. Tim was very happy. He was having a great time playing with his friends.
At the park, Tim saw a big tree. He wanted to climb it. He asked his mom, "Can I climb the tree?" His mom said, "Yes, we can climb the tree!"
Tim climbed the tree and reached up. He climbed the tree, but he was still too high. He tried and tried, but he could not reach the top. Then, he had an idea. He would climb the tree first.
Tim climbed up the tree. He climbed up and up. He climbed up the tree. He was very happy. But, he was not sure because he was stuck in the tree. He looked under the tree.
Then, something unexpected happened. A little bird flew down. The bird said, "Don't worry, Tim. I will help you." The bird was happy. It flew up high and up. Tim was not scared anymore. He learned that he could be brave and h...
<|endoftext|>


Anna and Ben were twins who liked to play with toys. One day, they found a big box in the yard. The box had many toys inside. Anna wanted to play with them.
"Look, Ben, I have a toy car!" Anna said. She took the car and ran to the box. She started to pull the car. She pushed and pulled it.
"No, I'm astron named Sam. I don't like this." Ben said. He felt sad.
"I don't like it. It's just a toy. But I like it. It's nice." Anna said. She liked Sam too.
"OK, Ben, I will share with you. But you have to be careful. I don't want to share. I want to play with you. I want to play with you. You are not my friend anymore. You are my friend. Can I play with you?" Ben asked.
"Of course, Ben. You can play with us. We can share and play together. We can share. We can play with you and the other things. You can play with us and the toys. We can be happy."
"Okay, Ben. Can we be friends?" Ben asked.
They played with the toys and made a big mess. They did not like it. They did not like
```

Since the dataset was so large, instead of simply loading the data from a URL, we decided to download it and stream it to tokenize. However, we quickly found that this was very slow. Furthermore, we found it interesting that the next token prediction/top 3 token prediction still seemed oddly low despite the output looking much, much better.

```
0, 24100, 1.6199465990066528
0, 24200, 1.5744138956069946
total loss: 1.5309447586536407, accuracy of next token prediction: 0.612030029296875%,
 accuracy that one of the top 3 predicted tokens is correct: 0.8000244140625
0, 24300, 1.6869858503341675
0, 24400, 1.6769109964370728
total loss: 1.5286414682865144, accuracy of next token prediction: 0.61173095703125%,
 accuracy that one of the top 3 predicted tokens is correct: 0.80089111328125
0, 24500, 1.5963815450668335
0, 24600, 1.6370590925216675
0, 24700, 1.5986320972442627
total loss: 1.5285784661769868, accuracy of next token prediction: 0.612445068359375%,
 accuracy that one of the top 3 predicted tokens is correct: 0.800653076171875
0, 24800, 1.6367319822311401
0, 24900, 1.611398458480835
total loss: 1.5254644691944121, accuracy of next token prediction: 0.61318359375%,
 accuracy that one of the top 3 predicted tokens is correct: 0.800714111328125
```

```
0, 19400, 1.7186824083328247
total loss: 1.5678027629852296, accuracy of next token prediction: 0.605975341796875%,
 accuracy that one of the top 3 predicted tokens is correct: 0.79444580078125
0, 19500, 1.6695902347564697
0, 19600, 1.6080902814865112
0, 19700, 1.564112901687622
total loss: 1.5641058921813964, accuracy of next token prediction: 0.605084228515625%,
 accuracy that one of the top 3 predicted tokens is correct: 0.79505615234375
0, 19800, 1.7289695739746094
0, 19900, 1.608230710029602
total loss: 1.5627849340438842, accuracy of next token prediction: 0.606683349609375%,
 accuracy that one of the top 3 predicted tokens is correct: 0.794683837890625
0, 20000, 1.6231377124786377
0, 20100, 1.6664772033691406
0, 20200, 1.670387864112854
total loss: 1.5602753639221192, accuracy of next token prediction: 0.606439208984375%,
 accuracy that one of the top 3 predicted tokens is correct: 0.796014404296875
0, 20300, 1.690801739692688
0, 20400, 1.5677520036697388
total loss: 1.5608687937259673, accuracy of next token prediction: 0.6066162109375%,
 accuracy that one of the top 3 predicted tokens is correct: 0.79534912109375
```

Inference:

```python
prompt = "Tom and Jane are friends. One day, Jane goes to Tom’s house. Tom has a big pot of soup. He wants to share it with Jane. “Jane, do you want some soup?” Tom asks. Yes, please. It looks yummy, Jane says. Tom pours some soup into two bowls. He gives one bowl to Jane. Jane takes a spoonful of soup, but then she makes a face. The soup is"
inference_loop(prompt, model, .7, 128, _)
```
```
Tom and Jane are friends. One day, Jane goes to Tom’s house. Tom has a big pot of soup. He wants to share it with Jane. “Jane, do you want some soup?” Tom asks. Yes, please. It looks yummy, Jane says. Tom pours some soup into two bowls. He gives one bowl to Jane. Jane takes a spoonful of soup, but then she makes a face. The soup is not good for my head. It is hot and sticky. It makes me feel bad.
Jane is sad. She says, “Ben, I don't want to share the soup. It is not fun. You can't have it.” But Ben does not listen. He says, “No, Jen, we can't. It is not nice. We can't share the soup. We will share and take turns. The soup is not good for us. It is not good for us. It is not yummy. It is good for our soup.”
Jen laughs. She says, "Ben, you are silly. You should share the soup with others. It is good for you."
She takes the
```

### Earlier versions

With our first character-level model, after running with batch size 64 and 10 epochs, we got a loss of 2.3587 at step 7800. It took 40+ minutes running on a v5e-1 TPU (no CUDA).

Improvements from v1: using CUDA, and changing the computationally heavy rerunning on every sequence, so instead of looking at every sequence we looked at sequences that differ by the context size so there's no more overlap (we later switched to random sampling like nanoGPT).

Making the first model generate 100 new tokens, we got this:

```
Shall I compare thee to a summer's day? Thou art more lovely and more temperate: Rough winds do shake the darling buds of May,
yomeroWh cot Of Inovey lld whoust isilche,
Gice th! O artensou, pr lthouscaf aced Cawik an trye,
Wh
```
