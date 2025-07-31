import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        self.device = torch.device('cpu')
        self.to(self.device)

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        
        # 토크나이저에서 padding token ID 가져오기
        pad_token_id = tokenizer.pad_token_id
        
        print(f"Starting Word2Vec training with {self.method} method")
        print(f"Corpus size: {len(corpus)} sentences")
        print(f"Window size: {self.window_size}, Learning rate: {lr}")
        
        for epoch in range(num_epochs):
            total_loss: float = 0.0
            num_batches: int = 0
            
            print(f"\nEpoch {epoch + 1}/{num_epochs} starting...")
            
            for sentence_idx, sentence in enumerate(corpus):
                tokens = tokenizer.encode(sentence, add_special_tokens=False)
                tokens = [token for token in tokens if token != pad_token_id]
                
                if len(tokens) < 2 * self.window_size + 1:
                    continue
                
                if self.method == "cbow":
                    loss = self._train_cbow(tokens, criterion, optimizer)
                else:  # skipgram
                    loss = self._train_skipgram(tokens, criterion, optimizer)
                
                total_loss += loss
                num_batches += 1
                
                # 1000문장마다 진행상황 출력
                if (sentence_idx + 1) % 50 == 0:
                    avg_loss_so_far = total_loss / num_batches
                    print(f"  Processed {sentence_idx + 1}/{len(corpus)} sentences, Avg Loss: {avg_loss_so_far:.4f}")
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1} completed! Average Loss: {avg_loss:.4f}")
                print(f"  Total batches processed: {num_batches}")

    def _train_cbow(
        self,
        tokens: list[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> float:
        # 구현하세요!
        batch_size = 64
        contexts: list[list[int]] = []
        targets: list[int] = []
        total_loss: float = 0.0
        num_batches_processed: int = 0
        
        # 모든 (context, target) 쌍을 미리 생성
        pairs: list[tuple[list[int], int]] = []
        for i in range(self.window_size, len(tokens) - self.window_size):
            # 주변 단어들을 context로 사용
            context_words: list[int] = []
            for j in range(-self.window_size, self.window_size + 1):
                if j != 0:  # target word 제외
                    context_words.append(tokens[i + j])
            
            target_word = tokens[i]
            pairs.append((context_words, target_word))
        
        # 배치 단위로 처리
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            if batch_pairs:
                batch_contexts = [pair[0] for pair in batch_pairs]
                batch_targets = [pair[1] for pair in batch_pairs]
                
                loss = self._train_cbow_batch(batch_contexts, batch_targets, criterion, optimizer)
                total_loss += loss
                num_batches_processed += 1
        
        return total_loss / num_batches_processed if num_batches_processed > 0 else 0.0

    def _train_cbow_batch(self, contexts: list[list[int]], targets: list[int], criterion: nn.CrossEntropyLoss, optimizer: Adam) -> float:
        # 배치 처리
        max_context_len = max(len(ctx) for ctx in contexts)
        
        # 패딩으로 배치 생성
        padded_contexts: list[list[int]] = []
        for ctx in contexts:
            padded = ctx + [ctx[0]] * (max_context_len - len(ctx))  # 패딩
            padded_contexts.append(padded)
        
        context_tensor = torch.tensor(padded_contexts, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        
        # Forward pass
        context_embeddings = self.embeddings(context_tensor)  # [batch, max_len, d_model]
        context_mean = context_embeddings.mean(dim=1)  # [batch, d_model]
        output = self.weight(context_mean)  # [batch, vocab_size]
        
        # Loss 계산 및 역전파
        loss = criterion(output, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def _train_skipgram(
        self,
        tokens: list[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> float:
        # 구현하세요!
        batch_size = 128
        targets: list[int] = []
        contexts: list[int] = []
        total_loss: float = 0.0
        num_batches_processed: int = 0
        
        # 모든 (target, context) 쌍을 미리 생성
        pairs: list[tuple[int, int]] = []
        for i in range(len(tokens)):
            target_word = tokens[i]
            # window 내의 주변 단어들을 찾기
            for j in range(max(0, i - self.window_size), min(len(tokens), i + self.window_size + 1)):
                if i != j:  # 자기 자신 제외
                    context_word = tokens[j]
                    pairs.append((target_word, context_word))
        
        # 배치 단위로 처리
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            if batch_pairs:
                batch_targets = [pair[0] for pair in batch_pairs]
                batch_contexts = [pair[1] for pair in batch_pairs]
                
                loss = self._train_skipgram_batch(batch_targets, batch_contexts, criterion, optimizer)
                total_loss += loss
                num_batches_processed += 1
        
        return total_loss / num_batches_processed if num_batches_processed > 0 else 0.0

    def _train_skipgram_batch(self, targets: list[int], contexts: list[int], criterion: nn.CrossEntropyLoss, optimizer: Adam) -> float:
        # 배치 처리
        target_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        context_tensor = torch.tensor(contexts, dtype=torch.long, device=self.device)
        
        # Forward pass
        target_embeddings = self.embeddings(target_tensor)  # [batch, d_model]
        output = self.weight(target_embeddings)  # [batch, vocab_size]
        
        # Loss 계산 및 역전파
        loss = criterion(output, context_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    # 구현하세요!
    def forward(self, x: LongTensor) -> Tensor:
        """Forward pass for inference"""
        embeddings = self.embeddings(x)
        return self.weight(embeddings)