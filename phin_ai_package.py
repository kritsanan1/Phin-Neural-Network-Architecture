"""
Phin Neural Network Architecture - Complete Package
Specialized architecture for Thai Phin instrument analysis and generation

This package provides a complete neural network solution for:
1. Phin music analysis and feature extraction
2. Pentatonic scale-aware music generation
3. Traditional Thai rhythm pattern recognition
4. Playing technique classification
5. Cultural music preservation and generation

Architecture Components:
- PhinPentatonicEmbedding: Custom embedding layer for pentatonic scales
- PhinRhythmAttention: Attention mechanism for rhythmic patterns
- PhinTechniqueEncoder: Encoder for Phin-specific playing techniques
- PhinScaleAwareDecoder: Decoder with scale/mode awareness
- PhinNeuralNetwork: Complete architecture combining all components

Usage:
    from phin_neural_network import create_phin_model
    from phin_data_preprocessing import create_phin_dataset
    from phin_training_framework import PhinTrainingFramework
    
    # Create model
    model = create_phin_model(vocab_size=128, embed_dim=256)
    
    # Process dataset
    preprocessor = create_phin_dataset("midi_folder", "dataset.json")
    
    # Train model
    trainer = PhinTrainingFramework(model)
    trainer.train(train_loader, val_loader, num_epochs=50)
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Specialized Neural Network Architecture for Thai Phin Instrument"

# Import main components
from phin_neural_network import (
    PhinPentatonicEmbedding,
    PhinRhythmAttention, 
    PhinTechniqueEncoder,
    PhinScaleAwareDecoder,
    PhinNeuralNetwork,
    create_phin_model
)

from phin_data_preprocessing import (
    PhinDataPreprocessor,
    create_phin_dataset
)

from phin_training_framework import (
    PhinDataset,
    PhinLossFunction,
    PhinTrainingFramework,
    evaluate_phin_model
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Specialized Neural Network Architecture for Thai Phin Instrument"

__all__ = [
    # Neural Network Components
    'PhinPentatonicEmbedding',
    'PhinRhythmAttention',
    'PhinTechniqueEncoder',
    'PhinScaleAwareDecoder',
    'PhinNeuralNetwork',
    'create_phin_model',
    
    # Data Processing
    'PhinDataPreprocessor',
    'create_phin_dataset',
    
    # Training
    'PhinDataset',
    'PhinLossFunction',
    'PhinTrainingFramework',
    'evaluate_phin_model'
]


class PhinAIModel:
    """High-level interface for Phin AI model"""
    
    def __init__(self, vocab_size: int = 128, embed_dim: int = 256, device: str = None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.preprocessor = None
        self.trainer = None
        
    def build_model(self) -> PhinNeuralNetwork:
        """Build the Phin neural network model"""
        self.model = create_phin_model(self.vocab_size, self.embed_dim)
        self.model.to(self.device)
        return self.model
    
    def preprocess_data(self, midi_folder: str, output_path: str, 
                       scale_type: str = 'A_minor', sequence_length: int = 64) -> PhinDataPreprocessor:
        """Preprocess MIDI dataset"""
        self.preprocessor = create_phin_dataset(
            midi_folder=midi_folder,
            output_path=output_path,
            scale_type=scale_type,
            sequence_length=sequence_length
        )
        return self.preprocessor
    
    def train_model(self, train_data, val_data, num_epochs: int = 50, 
                   save_path: str = None) -> dict:
        """Train the Phin model"""
        if self.model is None:
            self.build_model()
            
        self.trainer = PhinTrainingFramework(self.model, self.device)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False)
        
        # Train
        history = self.trainer.train(train_loader, val_loader, num_epochs, save_path)
        return history
    
    def generate_music(self, seed_notes: list, max_length: int = 128, 
                      temperature: float = 0.8, mode: int = 0) -> list:
        """Generate new Phin music"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Convert seed to tensor
        seed_tensor = torch.tensor(seed_notes, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                seed_notes=seed_tensor,
                max_length=max_length,
                temperature=temperature,
                mode=torch.tensor([mode], device=self.device)
            )
        
        return generated.squeeze().cpu().numpy().tolist()
    
    def analyze_midi(self, midi_path: str, scale_type: str = 'A_minor') -> dict:
        """Analyze MIDI file for Phin-specific features"""
        if self.preprocessor is None:
            self.preprocessor = PhinDataPreprocessor()
        
        features = self.preprocessor.process_midi_file(midi_path, scale_type)
        return features
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'device': self.device
        }, path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Build model if not exists
        if self.model is None:
            self.vocab_size = checkpoint.get('vocab_size', 128)
            self.embed_dim = checkpoint.get('embed_dim', 256)
            self.build_model()
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {path}")


# Convenience functions for quick usage
def quick_train(midi_folder: str, output_model_path: str, num_epochs: int = 50):
    """Quick training pipeline for Phin model"""
    
    print("Starting Phin model training...")
    
    # Create AI model
    phin_ai = PhinAIModel()
    
    # Preprocess data
    print("Preprocessing MIDI files...")
    phin_ai.preprocess_data(midi_folder, "phin_dataset.json")
    
    # Build model
    print("Building neural network...")
    phin_ai.build_model()
    
    # Train model (this is a simplified version - you'd need to load your data)
    print("Training model...")
    # phin_ai.train_model(train_data, val_data, num_epochs, output_model_path)
    
    print("Training pipeline ready!")
    return phin_ai


def generate_phin_music(model_path: str, seed_notes: list, max_length: int = 128) -> list:
    """Generate Phin music using trained model"""
    
    phin_ai = PhinAIModel()
    phin_ai.load_model(model_path)
    
    generated = phin_ai.generate_music(seed_notes, max_length)
    return generated


def analyze_phin_midi(midi_path: str, scale_type: str = 'A_minor') -> dict:
    """Analyze Phin MIDI file"""
    
    phin_ai = PhinAIModel()
    features = phin_ai.analyze_midi(midi_path, scale_type)
    return features


# Package information
print(f"Phin Neural Network Architecture v{__version__}")
print(f"Designed for Thai Phin instrument analysis and generation")
print(f"Features: Pentatonic scales, Traditional rhythms, Playing techniques")
print("")
print("Main components:")
print("- PhinNeuralNetwork: Complete architecture")
print("- PhinDataPreprocessor: MIDI processing")
print("- PhinTrainingFramework: Training pipeline")
print("- PhinAIModel: High-level interface")
print("")
print("Usage examples:")
print(">>> from phin_ai_package import PhinAIModel, quick_train")
print(">>> phin_ai = PhinAIModel()")
print(">>> phin_ai.build_model()")
print(">>> generated = phin_ai.generate_music([60, 62, 64], max_length=64)")