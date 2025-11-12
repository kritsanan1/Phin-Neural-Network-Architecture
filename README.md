# Phin Neural Network Architecture

## ภาษาไทย

### สถาปัตยกรรมเครือข่ายประสาทเฉพาะสำหรับเครื่องดนตรีไทย "พิณ"

สถาปัตยกรรมนี้ได้รับการออกแบบโดยเฉพาะสำหรับการวิเคราะห์และสร้างดนตรีของเครื่องดนตรีไทย "พิณ" โดยคำนึงถึงลักษณะเฉพาะของระบบเสียงแบบเพนแททอนิก (5 เสียง) และรูปแบบจังหวะดั้งเดิมของไทย

#### คุณสมบัติหลัก

🎵 **ระบบเพนแททอนิก (Pentatonic Scale System)**
- รองรับระบบ 5 เสียงแบบดั้งเดิมของพิณ
- คีย์หลัก: A minor pentatonic (A, C, D, E, G)
- รองรับโหมดต่างๆ ทั้งหมด 7 แบบ

🥁 **การรู้จำรูปแบบจังหวะไทย**
- รูปแบบจังหวะดั้งเดิม 8 แบบ เช่น สำฉันท์, ชังหว่า, โปงลาง
- การวิเคราะห์ความคล้ายคลึงของรูปแบบจังหวะ
- การประมาณค่า BPM อัตโนมัติ

🎨 **การรู้จำเทคนิคการเล่น**
- เทคนิคการเล่นพิณ 16 แบบ เช่น bend, slide, hammer-on, vibrato
- การตรวจจับจากการวิเคราะห์ MIDI
- การสร้างลำดับเทคนิคสำหรับการฝึกสอน

🧠 **สถาปัตยกรรมเครือข่ายประสาท**
- Custom embedding layer สำหรับระบบเพนแททอนิก
- Attention mechanism สำหรับรูปแบบจังหวะ
- Encoder-decoder ที่ตระหนักรู้เกี่ยวกับสเกลและโหมด
- Multi-task learning: การทำนายโน้ต, เทคนิค, จังหวะ, และโหมด

## English

### Specialized Neural Network Architecture for Thai "Phin" Instrument

This architecture is specifically designed for analyzing and generating music for the Thai "Phin" instrument, considering the unique characteristics of the pentatonic scale system (5 notes) and traditional Thai rhythmic patterns.

#### Key Features

🎵 **Pentatonic Scale System**
- Supports traditional 5-note system of Phin
- Primary key: A minor pentatonic (A, C, D, E, G)
- Supports all 7 modal variations

🥁 **Thai Rhythmic Pattern Recognition**
- 8 traditional rhythmic patterns such as Samchan, Changwa, Ponglang
- Rhythmic pattern similarity analysis
- Automatic BPM estimation

🎨 **Playing Technique Recognition**
- 16 Phin playing techniques such as bend, slide, hammer-on, vibrato
- Detection from MIDI analysis
- Technique sequence generation for training

🧠 **Neural Network Architecture**
- Custom embedding layer for pentatonic system
- Attention mechanism for rhythmic patterns
- Scale/mode-aware encoder-decoder
- Multi-task learning: note, technique, rhythm, and mode prediction

## Installation

```bash
# Install dependencies
pip install torch torchvision torchaudio
pip install pretty_midi music21 librosa
pip install numpy pandas matplotlib seaborn scikit-learn

# Clone or download the package files
# phin_neural_network.py
# phin_data_preprocessing.py  
# phin_training_framework.py
# phin_ai_package.py
```

## Quick Start

```python
from phin_ai_package import PhinAIModel, quick_train

# Create and train model
phin_ai = PhinAIModel()
phin_ai.build_model()

# Preprocess MIDI data
phin_ai.preprocess_data("midi_folder", "dataset.json")

# Generate music
seed_notes = [60, 62, 64, 65, 67]  # A minor pentatonic seed
generated = phin_ai.generate_music(seed_notes, max_length=128)

# Analyze MIDI file
features = phin_ai.analyze_midi("phin_music.mid", scale_type="A_minor")
```

## Architecture Details

### Core Components

1. **PhinPentatonicEmbedding**
   - Custom embedding layer with pentatonic bias
   - Position embeddings for sequence information
   - Supports all 12 MIDI note numbers with pentatonic emphasis

2. **PhinRhythmAttention**
   - Multi-head attention mechanism
   - Rhythm pattern recognition and bias
   - Supports 8 traditional Thai rhythmic patterns

3. **PhinTechniqueEncoder**
   - LSTM-based technique sequence processing
   - 16 different playing techniques
   - Technique embedding and classification

4. **PhinScaleAwareDecoder**
   - Mode-aware decoding (7 modes)
   - Pentatonic constraint enforcement
   - Scale-specific note probability adjustment

### Model Specifications

- **Input**: MIDI note sequences (128 vocab size)
- **Embedding Dimension**: 256 (configurable)
- **Attention Heads**: 8
- **Layers**: 6 transformer layers
- **Output**: Next note prediction, technique classification, rhythm analysis, mode prediction

### Loss Functions

- **Note Prediction**: Cross-entropy loss
- **Technique Classification**: Cross-entropy loss  
- **Rhythm Analysis**: MSE loss for pattern matching
- **Mode Prediction**: Cross-entropy loss
- **Pentatonic Constraint**: Custom penalty for non-pentatonic notes

## Dataset Structure

The system expects MIDI files with Phin music in A minor pentatonic scale. The preprocessing pipeline extracts:

- **Pentatonic Features**: Ratio of notes within pentatonic scale
- **Rhythmic Patterns**: 8-beat pattern classification
- **Playing Techniques**: 16 different technique types
- **Metadata**: Tempo, time signature, style information

Example dataset entry:
```json
{
  "metadata": {
    "lai_name": "ลายโปงลาง",
    "scale": "A_minor_pentatonic", 
    "tempo": 120,
    "time_signature": "4/4"
  },
  "notation": {
    "midi_file": "ponglang_01.mid",
    "notes": [60, 62, 64, 65, 67, 69, 71, 72],
    "techniques": [0, 1, 0, 2, 0, 1, 0, 3]
  }
}
```

## Training Process

1. **Data Preprocessing**: Convert MIDI files to training sequences
2. **Model Initialization**: Build neural network with specified architecture
3. **Training Loop**: Multi-task learning with custom loss function
4. **Validation**: Evaluate on held-out test set
5. **Generation**: Use trained model for music generation

## Performance Metrics

- **Note Accuracy**: Percentage of correctly predicted notes
- **Technique Accuracy**: Classification accuracy for playing techniques
- **Rhythm Similarity**: Pattern matching score for rhythmic analysis
- **Pentatonic Adherence**: Constraint satisfaction for pentatonic scale

## Applications

🎼 **Music Generation**: Create new Phin music in traditional style
📊 **Music Analysis**: Extract features from existing Phin recordings  
🎓 **Educational Tool**: Help students learn traditional patterns
💾 **Cultural Preservation**: Digitally preserve traditional music
🔄 **Fusion Music**: Combine Phin with other musical styles

## Traditional Thai Music Integration

The architecture specifically handles:

- **Lai (ลาย)**: Traditional melodic patterns
- **Thang (ทาง)**: Playing styles and techniques
- **Pitas (พิษ)**: Rhythmic cycles
- **Kratai (กระท่อม)**: Structural forms

Supported traditional patterns:
- ลายลำเพลิน (Lai Lam Phloen)
- ลายสุดสะแนน (Lai Sut Sa Nen)  
- ลายโปงลาง (Lai Ponglang)
- ลายภูไท (Lai Phu Thai)
- ลายเซิ้งบั้งไฟ (Lai Seng Bang Fai)

## Research and Development

This architecture is designed for:

1. **Cultural Musicology Research**: Analyze traditional music patterns
2. **AI Music Generation**: Create culturally appropriate music
3. **Music Education**: Interactive learning systems
4. **Digital Preservation**: Archive traditional performances
5. **Cross-cultural Fusion**: Combine with other musical traditions

## Future Enhancements

- **Audio-to-MIDI**: Direct audio processing
- **Real-time Generation**: Live performance support
- **Multi-instrument**: Expand to other Thai instruments
- **Style Transfer**: Convert between different regional styles
- **Interactive Learning**: Real-time feedback for students

## Citation

If you use this architecture in your research, please cite:

```bibtex
@software{phin_neural_network_2024,
  title={Phin Neural Network Architecture: Specialized AI for Thai Traditional Music},
  author={AI Assistant},
  year={2024},
  url={https://github.com/your-repo/phin-neural-network}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thai musicologists and cultural experts
- Traditional Phin masters and performers  
- Academic institutions studying Thai music
- Open source music processing libraries
- AI/ML research community

---

**Note**: This is a specialized architecture designed specifically for the Thai "Phin" instrument and its unique musical characteristics. For general music generation, consider using more general-purpose architectures like Music Transformer or MuseNet.