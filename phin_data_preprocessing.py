"""
Phin Data Preprocessing Pipeline
Specialized preprocessing for Thai Phin instrument data
Handles pentatonic scale analysis, rhythm pattern extraction, and technique recognition
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import pretty_midi
import librosa
import music21
# from midiutil import MIDIFile  # Optional - for MIDI generation
import json
import os
from pathlib import Path


class PhinDataPreprocessor:
    """Comprehensive data preprocessing for Phin instrument"""
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Phin-specific configurations
        self.pentatonic_scales = {
            'A_minor': [0, 3, 5, 7, 10],  # A, C, D, E, G
            'C_major': [0, 2, 4, 7, 9],   # C, D, E, G, A
            'D_minor': [2, 5, 7, 10, 0],  # D, F, G, A#, C
            'E_minor': [4, 7, 9, 0, 2],   # E, G, A, C, D
            'G_major': [7, 10, 0, 2, 4]    # G, A#, C, D, E
        }
        
        # Traditional Phin techniques mapping
        self.techniques = {
            'none': 0, 'bend': 1, 'slide': 2, 'hammer-on': 3, 'pull-off': 4,
            'vibrato': 5, 'mute': 6, 'harmonic': 7, 'tremolo': 8,
            'glissando': 9, 'trill': 10, 'mordent': 11, 'turn': 12,
            'grace-note': 13, 'appoggiatura': 14, 'acciaccatura': 15
        }
        
        # Traditional Thai rhythmic patterns (8-beat cycles)
        self.thai_rhythms = {
            'samchan': [1, 0, 1, 1, 0, 1, 1, 0],      # สำฉันท์
            'changwa': [1, 1, 0, 1, 0, 1, 1, 0],     # ชังหว่า
            'mong': [1, 0, 1, 0, 1, 0, 1, 0],       # มง
            'sabud': [1, 1, 1, 0, 1, 1, 0, 1],      # สะบัด
            'klon': [1, 0, 1, 1, 1, 0, 1, 1],       # กลอน
            'lai_thai': [1, 1, 0, 0, 1, 1, 0, 0],   # ลายไทย
            'ponglang': [1, 0, 0, 1, 1, 0, 0, 1],    # โปงลาง
            'samer': [1, 1, 1, 1, 0, 0, 0, 0]       # เสมอ
        }
        
        # Phin playing styles
        self.phin_styles = {
            'lai_lam_phloen': 'ลายลำเพลิน',
            'lai_sut_sa_nen': 'ลายสุดสะแนน', 
            'lai_ka_ten_kon': 'ลายกาเต้นก้อน',
            'lai_ponglang': 'ลายโปงลาง',
            'lai_phu_thai': 'ลายภูไท',
            'lai_seng_bang_fai': 'ลายเซิ้งบั้งไฟ',
            'lai_hae': 'ลายแห่'
        }
    
    def load_midi_file(self, midi_path: str) -> pretty_midi.PrettyMIDI:
        """Load MIDI file and validate Phin-specific content"""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            return midi_data
        except Exception as e:
            raise ValueError(f"Failed to load MIDI file {midi_path}: {str(e)}")
    
    def extract_pentatonic_features(self, midi_data: pretty_midi.PrettyMIDI, 
                                scale_type: str = 'A_minor') -> Dict:
        """Extract pentatonic scale features from MIDI data"""
        
        # Get all notes
        all_notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    all_notes.append(note.pitch)
        
        if not all_notes:
            return {'pentatonic_ratio': 0.0, 'out_of_scale_notes': []}
        
        # Get pentatonic scale
        pentatonic_scale = self.pentatonic_scales.get(scale_type, self.pentatonic_scales['A_minor'])
        
        # Count pentatonic vs non-pentatonic notes
        pentatonic_count = 0
        total_notes = len(all_notes)
        out_of_scale = []
        
        for note in all_notes:
            # Convert to scale degree (mod 12)
            note_mod = note % 12
            if note_mod in pentatonic_scale:
                pentatonic_count += 1
            else:
                out_of_scale.append(note)
        
        pentatonic_ratio = pentatonic_count / total_notes if total_notes > 0 else 0
        
        return {
            'pentatonic_ratio': pentatonic_ratio,
            'out_of_scale_notes': out_of_scale,
            'total_notes': total_notes,
            'pentatonic_notes': pentatonic_count,
            'scale_type': scale_type
        }
    
    def extract_rhythmic_patterns(self, midi_data: pretty_midi.PrettyMIDI) -> Dict:
        """Extract and classify rhythmic patterns"""
        
        # Get note durations and onset times
        durations = []
        onset_times = []
        
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    durations.append(note.end - note.start)
                    onset_times.append(note.start)
        
        if not durations:
            return {'rhythm_class': 'unknown', 'pattern_similarity': {}, 'tempo': 0}
        
        # Calculate tempo
        tempo = self.estimate_tempo(onset_times)
        
        # Quantize durations to 8th note grid
        quantized_pattern = self.quantize_rhythm(durations, tempo)
        
        # Find closest rhythmic pattern
        pattern_similarity = {}
        for pattern_name, pattern in self.thai_rhythms.items():
            similarity = self.calculate_pattern_similarity(quantized_pattern, pattern)
            pattern_similarity[pattern_name] = similarity
        
        # Get most similar pattern
        best_pattern = max(pattern_similarity, key=pattern_similarity.get)
        
        return {
            'rhythm_class': best_pattern,
            'pattern_similarity': pattern_similarity,
            'tempo': tempo,
            'quantized_pattern': quantized_pattern,
            'original_durations': durations
        }
    
    def estimate_tempo(self, onset_times: List[float]) -> float:
        """Estimate tempo from onset times"""
        if len(onset_times) < 2:
            return 120.0  # Default tempo
        
        # Calculate inter-onset intervals
        iois = []
        sorted_times = sorted(onset_times)
        for i in range(1, len(sorted_times)):
            iois.append(sorted_times[i] - sorted_times[i-1])
        
        if not iois:
            return 120.0
        
        # Estimate tempo (BPM)
        avg_ioi = np.mean(iois)
        tempo = 60.0 / avg_ioi if avg_ioi > 0 else 120.0
        
        # Clamp to reasonable range
        return np.clip(tempo, 60.0, 200.0)
    
    def quantize_rhythm(self, durations: List[float], tempo: float) -> List[int]:
        """Quantize durations to 8-beat pattern"""
        beat_duration = 60.0 / tempo  # Duration of one beat in seconds
        
        # Convert durations to beat units
        beat_units = [duration / beat_duration for duration in durations]
        
        # Quantize to 8th note grid (0.5 beat units)
        quantized = []
        for beat in beat_units:
            quantized_beat = int(round(beat * 2)) / 2  # Round to nearest 8th note
            if quantized_beat > 0:
                quantized.append(min(quantized_beat, 2.0))  # Max 2 beats
        
        # Convert to 8-beat pattern (binary)
        pattern_length = 8
        pattern = [0] * pattern_length
        
        if quantized:
            avg_duration = np.mean(quantized)
            # Map longer durations to pattern positions
            for i, duration in enumerate(quantized[:pattern_length]):
                if duration >= avg_duration * 0.7:
                    pattern[i % pattern_length] = 1
        
        return pattern
    
    def calculate_pattern_similarity(self, pattern1: List[int], pattern2: List[int]) -> float:
        """Calculate similarity between two rhythmic patterns"""
        if len(pattern1) != len(pattern2):
            return 0.0
        
        matches = sum(1 for a, b in zip(pattern1, pattern2) if a == b)
        return matches / len(pattern1)
    
    def extract_technique_features(self, midi_data: pretty_midi.PrettyMIDI) -> Dict:
        """Extract playing technique features from MIDI data"""
        
        techniques_detected = []
        
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                notes = sorted(instrument.notes, key=lambda x: x.start)
                
                for i, note in enumerate(notes):
                    technique = 'none'
                    
                    # Detect techniques based on MIDI properties
                    # Bend: pitch bend messages
                    if hasattr(instrument, 'pitch_bends') and instrument.pitch_bends:
                        for pb in instrument.pitch_bends:
                            if abs(pb.pitch) > 0.1:  # Significant pitch bend
                                technique = 'bend'
                                break
                    
                    # Slide: consecutive notes with small intervals
                    if i > 0:
                        prev_note = notes[i-1]
                        interval = abs(note.pitch - prev_note.pitch)
                        time_gap = note.start - prev_note.end
                        
                        if interval <= 2 and time_gap < 0.1:  # Small interval, fast transition
                            technique = 'slide'
                        elif interval <= 4 and prev_note.end > note.start:  # Overlapping notes
                            if note.pitch > prev_note.pitch:
                                technique = 'hammer-on'
                            else:
                                technique = 'pull-off'
                    
                    # Vibrato: based on note modulation (simplified)
                    if note.end - note.start > 0.5:  # Long sustained note
                        technique = 'vibrato'
                    
                    # Tremolo: rapid repetition
                    if i > 1:
                        prev_prev_note = notes[i-2]
                        if (abs(note.pitch - prev_note.pitch) <= 1 and 
                            abs(prev_note.pitch - prev_prev_note.pitch) <= 1 and
                            note.start - prev_note.start < 0.2):
                            technique = 'tremolo'
                    
                    techniques_detected.append(self.techniques.get(technique, 0))
        
        # Count technique frequencies
        technique_counts = {}
        for tech in techniques_detected:
            technique_counts[tech] = technique_counts.get(tech, 0) + 1
        
        return {
            'techniques_sequence': techniques_detected,
            'technique_counts': technique_counts,
            'dominant_technique': max(set(techniques_detected), key=techniques_detected.count) if techniques_detected else 0
        }
    
    def process_midi_file(self, midi_path: str, scale_type: str = 'A_minor') -> Dict:
        """Process a single MIDI file and extract all features"""
        
        midi_data = self.load_midi_file(midi_path)
        
        # Extract all features
        pentatonic_features = self.extract_pentatonic_features(midi_data, scale_type)
        rhythmic_features = self.extract_rhythmic_patterns(midi_data)
        technique_features = self.extract_technique_features(midi_data)
        
        # Get file info
        file_stats = {
            'filename': os.path.basename(midi_path),
            'file_size': os.path.getsize(midi_path),
            'duration': midi_data.get_end_time(),
            'num_instruments': len(midi_data.instruments)
        }
        
        return {
            'file_info': file_stats,
            'pentatonic': pentatonic_features,
            'rhythm': rhythmic_features,
            'techniques': technique_features,
            'scale_type': scale_type
        }
    
    def create_training_sequence(self, features: Dict, sequence_length: int = 64) -> Dict:
        """Create training sequences from extracted features"""
        
        # Note sequence
        notes = features.get('notes', [60] * sequence_length)  # Default to middle C
        
        # Technique sequence
        techniques = features.get('techniques', {}).get('techniques_sequence', [0] * sequence_length)
        
        # Rhythm pattern
        rhythm_pattern = features.get('rhythm', {}).get('quantized_pattern', [0] * 8)
        
        # Scale/mode
        scale_type = features.get('scale_type', 'A_minor')
        mode_id = list(self.pentatonic_scales.keys()).index(scale_type) if scale_type in self.pentatonic_scales else 0
        
        # Pad or truncate sequences
        notes = self.pad_or_truncate(notes, sequence_length)
        techniques = self.pad_or_truncate(techniques, sequence_length)
        
        return {
            'notes': torch.tensor(notes, dtype=torch.long),
            'techniques': torch.tensor(techniques, dtype=torch.long),
            'rhythm_pattern': torch.tensor(rhythm_pattern, dtype=torch.float32),
            'mode': torch.tensor(mode_id, dtype=torch.long),
            'metadata': features
        }
    
    def pad_or_truncate(self, sequence: List, target_length: int) -> List:
        """Pad or truncate sequence to target length"""
        if len(sequence) > target_length:
            return sequence[:target_length]
        elif len(sequence) < target_length:
            padding = [sequence[-1] if sequence else 0] * (target_length - len(sequence))
            return sequence + padding
        return sequence
    
    def process_dataset(self, midi_folder: str, output_path: str, 
                       scale_type: str = 'A_minor', sequence_length: int = 64) -> Dict:
        """Process entire dataset of MIDI files"""
        
        midi_files = list(Path(midi_folder).glob('*.mid')) + list(Path(midi_folder).glob('*.midi'))
        
        processed_data = []
        failed_files = []
        
        print(f"Processing {len(midi_files)} MIDI files...")
        
        for i, midi_file in enumerate(midi_files):
            try:
                print(f"Processing {i+1}/{len(midi_files)}: {midi_file.name}")
                
                # Extract features
                features = self.process_midi_file(str(midi_file), scale_type)
                
                # Create training sequence
                training_seq = self.create_training_sequence(features, sequence_length)
                
                processed_data.append({
                    'filename': midi_file.name,
                    'features': features,
                    'training_sequence': training_seq
                })
                
            except Exception as e:
                print(f"Failed to process {midi_file.name}: {str(e)}")
                failed_files.append({
                    'filename': midi_file.name,
                    'error': str(e)
                })
                continue
        
        # Save processed data
        dataset_info = {
            'total_files': len(midi_files),
            'successful_files': len(processed_data),
            'failed_files': len(failed_files),
            'failed_details': failed_files,
            'scale_type': scale_type,
            'sequence_length': sequence_length
        }
        
        # Save to disk
        output_data = {
            'dataset_info': dataset_info,
            'processed_data': processed_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Convert tensors to lists for JSON serialization
            json_data = self._convert_tensors_to_lists(output_data)
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset processing complete!")
        print(f"Successful: {len(processed_data)}/{len(midi_files)} files")
        print(f"Output saved to: {output_path}")
        
        return dataset_info
    
    def _convert_tensors_to_lists(self, data):
        """Recursively convert PyTorch tensors to lists for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._convert_tensors_to_lists(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_tensors_to_lists(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.tolist()
        else:
            return data


def create_phin_dataset(midi_folder: str, output_path: str, 
                       scale_type: str = 'A_minor', sequence_length: int = 64) -> PhinDataPreprocessor:
    """Factory function to create and process Phin dataset"""
    
    preprocessor = PhinDataPreprocessor()
    
    # Process dataset
    dataset_info = preprocessor.process_dataset(
        midi_folder=midi_folder,
        output_path=output_path,
        scale_type=scale_type,
        sequence_length=sequence_length
    )
    
    return preprocessor


# Example usage
if __name__ == "__main__":
    # Example: Process a folder of Phin MIDI files
    midi_folder = "path/to/phin/midi/files"
    output_path = "phin_dataset.json"
    
    # Create preprocessor and process dataset
    preprocessor = create_phin_dataset(
        midi_folder=midi_folder,
        output_path=output_path,
        scale_type='A_minor',
        sequence_length=64
    )
    
    print("Phin dataset preprocessing complete!")
    print(f"Dataset saved to: {output_path}")