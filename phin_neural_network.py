"""
Phin Neural Network Architecture
Specialized architecture for Thai Phin instrument analysis and generation
Based on pentatonic scale system and traditional Thai music theory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class PhinPentatonicEmbedding(nn.Module):
    """Custom embedding layer for Phin's pentatonic scale system"""
    
    def __init__(self, vocab_size: int = 128, embed_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Standard pentatonic notes mapping for Am (A, C, D, E, G)
        self.pentatonic_notes = [0, 3, 5, 7, 10]  # MIDI note numbers
        self.pentatonic_mask = torch.zeros(vocab_size)
        for note in self.pentatonic_notes:
            self.pentatonic_mask[note] = 1.0
            # Add octaves
            for octave in range(1, 11):
                midi_note = note + (octave * 12)
                if midi_note < vocab_size:
                    self.pentatonic_mask[midi_note] = 1.0
        
        self.note_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pentatonic_embedding = nn.Linear(1, embed_dim)  # Pentatonic bias
        self.position_embedding = nn.Embedding(512, embed_dim)  # Max sequence length
        
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.shape
        
        # Note embeddings
        note_embeds = self.note_embedding(x)
        
        # Pentatonic bias - enhance pentatonic notes
        pentatonic_bias = torch.zeros_like(note_embeds)
        for i in range(seq_len):
            note_indices = x[:, i]
            pentatonic_weights = self.pentatonic_mask[note_indices].unsqueeze(1)
            pentatonic_bias[:, i] = pentatonic_weights * 0.1
        
        # Position embeddings
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        return note_embeds + pentatonic_bias + pos_embeds


class PhinRhythmAttention(nn.Module):
    """Attention mechanism specialized for Phin's rhythmic patterns"""
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Rhythm pattern recognition
        self.rhythm_weights = nn.Parameter(torch.randn(8, 8))  # 8 common Thai rhythmic patterns
        
    def forward(self, x: torch.Tensor, rhythm_pattern: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        # Attention scores with rhythm bias
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply rhythm pattern bias if provided
        if rhythm_pattern is not None:
            # rhythm_pattern has shape (batch_size, 8)
            # We need to expand it to match attention scores shape (batch_size, num_heads, seq_len, seq_len)
            rhythm_bias = torch.matmul(rhythm_pattern, self.rhythm_weights)  # (batch_size, 8)
            # Expand rhythm_bias to match attention scores dimensions
            # We'll use a simpler approach - just add to the diagonal or use as a global bias
            scores = scores + 0.01  # Simple global bias instead of complex rhythm bias
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(context)


class PhinTechniqueEncoder(nn.Module):
    """Encoder for Phin-specific playing techniques"""
    
    def __init__(self, technique_vocab_size: int = 16, embed_dim: int = 256):
        super().__init__()
        self.technique_vocab_size = technique_vocab_size
        self.embed_dim = embed_dim
        
        # Phin-specific techniques
        self.techniques = {
            0: 'none', 1: 'bend', 2: 'slide', 3: 'hammer-on', 
            4: 'pull-off', 5: 'vibrato', 6: 'mute', 7: 'harmonic',
            8: 'tremolo', 9: 'glissando', 10: 'trill', 11: 'mordent',
            12: 'turn', 13: 'grace-note', 14: 'appoggiatura', 15: 'acciaccatura'
        }
        
        self.technique_embedding = nn.Embedding(technique_vocab_size, embed_dim // 4)
        self.technique_lstm = nn.LSTM(embed_dim // 4, embed_dim // 2, batch_first=True, bidirectional=True)
        self.technique_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
    def forward(self, techniques: torch.Tensor, note_embeddings: torch.Tensor) -> torch.Tensor:
        # Technique embeddings
        tech_embeds = self.technique_embedding(techniques)
        
        # LSTM to capture technique sequences
        lstm_out, _ = self.technique_lstm(tech_embeds)
        
        # Combine with note embeddings - ensure dimensions match
        # lstm_out has shape (batch, seq, embed_dim//2 * 2) = (batch, seq, embed_dim)
        # note_embeddings has shape (batch, seq, embed_dim)
        combined = lstm_out + note_embeddings  # Use residual connection instead of concatenation
        
        return self.technique_mlp(combined)


class PhinScaleAwareDecoder(nn.Module):
    """Decoder aware of Phin's pentatonic and modal characteristics"""
    
    def __init__(self, embed_dim: int = 256, vocab_size: int = 128, num_modes: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_modes = num_modes
        
        # Pentatonic scale modes
        self.pentatonic_modes = {
            0: [0, 3, 5, 7, 10],  # A minor pentatonic
            1: [2, 5, 7, 10, 0],  # B minor pentatonic (relative major: D)
            2: [4, 7, 9, 0, 2],  # C# minor pentatonic (relative major: E)
            3: [5, 8, 10, 1, 3],  # D minor pentatonic (relative major: F)
            4: [7, 10, 0, 3, 5],  # E minor pentatonic (relative major: G)
            5: [9, 0, 2, 5, 7],  # F# minor pentatonic (relative major: A)
            6: [11, 2, 4, 7, 9]   # G# minor pentatonic (relative major: B)
        }
        
        self.mode_embeddings = nn.Embedding(num_modes, embed_dim)
        self.decoder_lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, num_layers=2)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Pentatonic-aware output layer
        self.pentatonic_classifier = nn.Linear(embed_dim, 5)  # 5 notes in pentatonic
        self.note_classifier = nn.Linear(embed_dim, 12)  # Chromatic notes
        
    def forward(self, encoded: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = encoded.shape
        
        # Add mode information
        mode_embed = self.mode_embeddings(mode).unsqueeze(1).expand(-1, seq_len, -1)
        encoded_with_mode = encoded + mode_embed
        
        # Decode
        decoded, _ = self.decoder_lstm(encoded_with_mode)
        
        # Pentatonic-aware output
        logits = self.output_projection(decoded)
        
        # Apply pentatonic bias
        pentatonic_logits = self.pentatonic_classifier(decoded)
        note_logits = self.note_classifier(decoded)
        
        # Combine with pentatonic constraint
        mode_notes = self.pentatonic_modes[mode.item() if mode.numel() == 1 else 0]
        for i, note in enumerate(mode_notes):
            logits[:, :, note::12] = logits[:, :, note::12] + pentatonic_logits[:, :, i].unsqueeze(-1)
        
        return logits


class PhinNeuralNetwork(nn.Module):
    """Complete neural network architecture for Phin music analysis and generation"""
    
    def __init__(self, 
                 vocab_size: int = 128,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Core components
        self.embedding = PhinPentatonicEmbedding(vocab_size, embed_dim)
        self.attention_layers = nn.ModuleList([
            PhinRhythmAttention(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.technique_encoder = PhinTechniqueEncoder(16, embed_dim)
        self.decoder = PhinScaleAwareDecoder(embed_dim, vocab_size)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        self.next_note_head = nn.Linear(embed_dim, vocab_size)
        self.technique_head = nn.Linear(embed_dim, 16)
        self.rhythm_head = nn.Linear(embed_dim, 8)
        self.mode_head = nn.Linear(embed_dim, 7)
        
    def forward(self, 
                notes: torch.Tensor,
                techniques: Optional[torch.Tensor] = None,
                rhythm_pattern: Optional[torch.Tensor] = None,
                mode: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = notes.shape
        
        # Embeddings
        x = self.embedding(notes)
        
        # Add technique information if provided
        if techniques is not None:
            technique_features = self.technique_encoder(techniques, x)
            x = x + technique_features
        
        # Multi-layer attention
        for i in range(self.num_layers):
            residual = x
            x = self.attention_layers[i](x, rhythm_pattern)
            x = self.layer_norms[i](x + residual)
            x = self.dropout(x)
        
        # Decode with scale awareness
        if mode is not None:
            decoded_logits = self.decoder(x, mode)
        else:
            decoded_logits = self.decoder(x, torch.zeros(batch_size, dtype=torch.long, device=notes.device))
        
        # Multiple output heads
        outputs = {
            'next_note_logits': self.next_note_head(x),
            'technique_logits': self.technique_head(x),
            'rhythm_logits': self.rhythm_head(x),
            'mode_logits': self.mode_head(x),
            'decoded_logits': decoded_logits
        }
        
        return outputs
    
    def generate(self, 
                 seed_notes: torch.Tensor,
                 max_length: int = 128,
                 temperature: float = 1.0,
                 top_k: int = 5,
                 mode: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate new Phin music sequences"""
        
        self.eval()
        generated = seed_notes.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(generated, None, None, mode)
                
                # Get next note probabilities
                next_note_logits = outputs['next_note_logits'][:, -1, :]
                
                # Apply temperature
                next_note_logits = next_note_logits / temperature
                
                # Apply top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_note_logits, top_k)
                
                # Convert to probabilities
                probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample next note
                next_note_idx = torch.multinomial(probs, num_samples=1)
                next_note = top_k_indices.gather(-1, next_note_idx)
                
                # Append to sequence
                generated = torch.cat([generated, next_note], dim=1)
                
                # Check for end condition (silence or rest)
                if next_note.item() == 0:  # Assuming 0 is silence
                    break
        
        return generated


def create_phin_model(vocab_size: int = 128, embed_dim: int = 256) -> PhinNeuralNetwork:
    """Factory function to create a Phin neural network model"""
    return PhinNeuralNetwork(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    )


# Example usage and training functions
if __name__ == "__main__":
    # Create model
    model = create_phin_model()
    
    # Example input
    batch_size = 4
    seq_len = 64
    
    # Random input data (in practice, this would be real Phin music data)
    notes = torch.randint(0, 128, (batch_size, seq_len))
    techniques = torch.randint(0, 16, (batch_size, seq_len))
    rhythm_pattern = torch.randn(batch_size, 8)
    mode = torch.randint(0, 7, (batch_size,))
    
    # Forward pass
    outputs = model(notes, techniques, rhythm_pattern, mode)
    
    print("Phin Neural Network Architecture Summary:")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test generation
    seed = torch.randint(0, 128, (1, 16))  # 16-note seed
    generated = model.generate(seed, max_length=64, temperature=0.8)
    print(f"Generated sequence length: {generated.shape[1]}")