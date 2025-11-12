"""
Phin Training and Evaluation Framework
Complete training pipeline for the Phin neural network architecture
Includes loss functions, metrics, and evaluation methods specific to Thai music
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import our custom modules
from phin_neural_network import PhinNeuralNetwork, create_phin_model
from phin_data_preprocessing import PhinDataPreprocessor


class PhinDataset(Dataset):
    """Dataset class for Phin music sequences"""
    
    def __init__(self, data: List[Dict], sequence_length: int = 64):
        self.data = data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract training sequences
        notes = item['training_sequence']['notes']
        techniques = item['training_sequence']['techniques']
        rhythm_pattern = item['training_sequence']['rhythm_pattern']
        mode = item['training_sequence']['mode']
        
        # Create input and target sequences (shift by 1 for next note prediction)
        input_notes = notes[:-1]
        target_notes = notes[1:]
        
        input_techniques = techniques[:-1]
        target_techniques = techniques[1:]
        
        return {
            'input_notes': input_notes,
            'target_notes': target_notes,
            'input_techniques': input_techniques,
            'target_techniques': target_techniques,
            'rhythm_pattern': rhythm_pattern,
            'mode': mode
        }


class PhinLossFunction(nn.Module):
    """Custom loss function for Phin music generation"""
    
    def __init__(self, 
                 note_weight: float = 1.0,
                 technique_weight: float = 0.5,
                 rhythm_weight: float = 0.3,
                 mode_weight: float = 0.2,
                 pentatonic_weight: float = 0.8):
        super().__init__()
        self.note_weight = note_weight
        self.technique_weight = technique_weight
        self.rhythm_weight = rhythm_weight
        self.mode_weight = mode_weight
        self.pentatonic_weight = pentatonic_weight
        
        # Loss functions
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """Calculate combined loss with Phin-specific constraints"""
        
        losses = {}
        
        # Note prediction loss
        note_loss = self.cross_entropy(
            predictions['next_note_logits'].view(-1, predictions['next_note_logits'].size(-1)),
            targets['target_notes'].view(-1)
        )
        losses['note_loss'] = note_loss
        
        # Technique prediction loss
        technique_loss = self.cross_entropy(
            predictions['technique_logits'].view(-1, predictions['technique_logits'].size(-1)),
            targets['target_techniques'].view(-1)
        )
        losses['technique_loss'] = technique_loss
        
        # Rhythm pattern loss
        rhythm_target = targets['rhythm_pattern'].expand(predictions['rhythm_logits'].size(0), -1)
        rhythm_loss = self.mse(predictions['rhythm_logits'], rhythm_target)
        losses['rhythm_loss'] = rhythm_loss
        
        # Mode prediction loss
        mode_loss = self.cross_entropy(predictions['mode_logits'], targets['mode'])
        losses['mode_loss'] = mode_loss
        
        # Pentatonic constraint loss
        pentatonic_loss = self.pentatonic_constraint_loss(predictions, targets)
        losses['pentatonic_loss'] = pentatonic_loss
        
        # Combined loss
        total_loss = (
            self.note_weight * note_loss +
            self.technique_weight * technique_loss +
            self.rhythm_weight * rhythm_loss +
            self.mode_weight * mode_loss +
            self.pentatonic_weight * pentatonic_loss
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def pentatonic_constraint_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """Penalize predictions that violate pentatonic scale constraints"""
        
        # Get note predictions
        note_logits = predictions['next_note_logits']
        
        # Create pentatonic mask (A minor pentatonic: A, C, D, E, G)
        pentatonic_notes = [0, 3, 5, 7, 10]  # MIDI note numbers mod 12
        pentatonic_mask = torch.zeros_like(note_logits)
        
        for note in pentatonic_notes:
            for octave in range(11):  # 11 octaves
                midi_note = note + (octave * 12)
                if midi_note < note_logits.size(-1):
                    pentatonic_mask[:, :, midi_note] = 1.0
        
        # Penalize non-pentatonic notes
        non_pentatonic_logits = note_logits * (1 - pentatonic_mask)
        pentatonic_penalty = torch.mean(torch.abs(non_pentatonic_logits))
        
        return pentatonic_penalty


class PhinTrainingFramework:
    """Complete training framework for Phin neural network"""
    
    def __init__(self, model: PhinNeuralNetwork, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.loss_function = PhinLossFunction()
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'note_accuracy': [],
            'technique_accuracy': [],
            'learning_rates': []
        }
        
    def setup_optimizer(self, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total_loss': 0.0, 'note_loss': 0.0, 'technique_loss': 0.0,
                   'rhythm_loss': 0.0, 'mode_loss': 0.0, 'pentatonic_loss': 0.0}
        
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(batch['input_notes'], batch['input_techniques'])
            
            # Calculate losses
            losses = self.loss_function(predictions, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{num_batches}, Loss: {losses['total_loss'].item():.4f}")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {'total_loss': 0.0, 'note_loss': 0.0, 'technique_loss': 0.0,
                  'rhythm_loss': 0.0, 'mode_loss': 0.0, 'pentatonic_loss': 0.0}
        
        note_predictions = []
        note_targets = []
        technique_predictions = []
        technique_targets = []
        
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch['input_notes'], batch['input_techniques'])
                
                # Calculate losses
                losses = self.loss_function(predictions, batch)
                
                # Accumulate losses
                for key in val_losses:
                    val_losses[key] += losses[key].item()
                
                # Collect predictions for accuracy calculation
                note_pred = predictions['next_note_logits'].argmax(dim=-1)
                note_predictions.extend(note_pred.view(-1).cpu().numpy())
                note_targets.extend(batch['target_notes'].view(-1).cpu().numpy())
                
                technique_pred = predictions['technique_logits'].argmax(dim=-1)
                technique_predictions.extend(technique_pred.view(-1).cpu().numpy())
                technique_targets.extend(batch['target_techniques'].view(-1).cpu().numpy())
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Calculate accuracies
        note_accuracy = accuracy_score(note_targets, note_predictions)
        technique_accuracy = accuracy_score(technique_targets, technique_predictions)
        
        val_losses['note_accuracy'] = note_accuracy
        val_losses['technique_accuracy'] = technique_accuracy
        
        return val_losses
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              num_epochs: int = 50, save_path: Optional[str] = None) -> Dict:
        """Complete training loop"""
        
        if self.optimizer is None:
            self.setup_optimizer()
        
        best_val_loss = float('inf')
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_losses = self.train_epoch(train_dataloader)
            
            # Validate
            val_losses = self.validate(val_dataloader)
            
            # Update learning rate
            self.scheduler.step(val_losses['total_loss'])
            
            # Record history
            self.training_history['train_losses'].append(train_losses['total_loss'])
            self.training_history['val_losses'].append(val_losses['total_loss'])
            self.training_history['note_accuracy'].append(val_losses['note_accuracy'])
            self.training_history['technique_accuracy'].append(val_losses['technique_accuracy'])
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {val_losses['total_loss']:.4f}, Note Acc: {val_losses['note_accuracy']:.3f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_losses['total_loss'] < best_val_loss and save_path:
                best_val_loss = val_losses['total_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_losses['total_loss'],
                    'note_accuracy': val_losses['note_accuracy']
                }, save_path)
                print(f"  Best model saved (val_loss: {val_losses['total_loss']:.4f})")
        
        print("Training completed!")
        return self.training_history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Phin Neural Network Training History', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_losses'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_losses'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.training_history['note_accuracy'], label='Note Accuracy')
        axes[0, 1].plot(self.training_history['technique_accuracy'], label='Technique Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.training_history['learning_rates'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss components
        if len(self.training_history['train_losses']) > 0:
            # This would require storing individual loss components during training
            axes[1, 1].text(0.1, 0.5, 'Loss Components\n(Detailed tracking\nrequired for\nfull breakdown)',
                          transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Loss Components')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()


def evaluate_phin_model(model: PhinNeuralNetwork, test_dataloader: DataLoader, 
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """Comprehensive evaluation of trained Phin model"""
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_technique_predictions = []
    all_technique_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = model(batch['input_notes'], batch['input_techniques'])
            
            # Collect predictions
            note_pred = predictions['next_note_logits'].argmax(dim=-1)
            technique_pred = predictions['technique_logits'].argmax(dim=-1)
            
            all_predictions.extend(note_pred.view(-1).cpu().numpy())
            all_targets.extend(batch['target_notes'].view(-1).cpu().numpy())
            
            all_technique_predictions.extend(technique_pred.view(-1).cpu().numpy())
            all_technique_targets.extend(batch['target_techniques'].view(-1).cpu().numpy())
    
    # Calculate metrics
    note_accuracy = accuracy_score(all_targets, all_predictions)
    technique_accuracy = accuracy_score(all_technique_targets, all_technique_predictions)
    
    # Detailed classification reports
    note_report = classification_report(all_targets, all_predictions, output_dict=True)
    technique_report = classification_report(all_technique_targets, all_technique_predictions, output_dict=True)
    
    results = {
        'note_accuracy': note_accuracy,
        'technique_accuracy': technique_accuracy,
        'note_classification_report': note_report,
        'technique_classification_report': technique_report,
        'total_predictions': len(all_predictions),
        'note_predictions': all_predictions,
        'note_targets': all_targets,
        'technique_predictions': all_technique_predictions,
        'technique_targets': all_technique_targets
    }
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("Phin Neural Network Training Framework")
    print("=" * 50)
    
    # Create model
    model = create_phin_model(vocab_size=128, embed_dim=256)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Example training setup (would need real data)
    # This is just a demonstration of the API
    
    # Create dummy data for demonstration
    dummy_data = []
    for i in range(100):
        dummy_item = {
            'training_sequence': {
                'notes': torch.randint(0, 128, (64,)),
                'techniques': torch.randint(0, 16, (64,)),
                'rhythm_pattern': torch.randn(8),
                'mode': torch.tensor(0)
            }
        }
        dummy_data.append(dummy_item)
    
    # Create dataset and dataloaders
    dataset = PhinDataset(dummy_data)
    
    # Split data
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    # Create training framework
    trainer = PhinTrainingFramework(model)
    
    print("Training framework ready!")
    print("To train with real data, you would call:")
    print("trainer.train(train_loader, val_loader, num_epochs=50, save_path='best_model.pt')")