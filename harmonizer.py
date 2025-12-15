"""
================================================
Neural Harmonization Engine: TTBB Vocal Arranger
================================================
A hybrid AI system combining:
1. Probabilistic chord prediction (learned from Bach chorales)
2. Constraint satisfaction for voice leading
3. Heuristic search for optimal harmonizations

This represents a classic AI approach: learning statistical patterns from data
combined with rule-based reasoning and search.
"""

import music21
from music21 import corpus, chord, note, stream, pitch, interval, key, meter
from collections import defaultdict
import random
import json
import os

# ============================================================================
# TTBB VOICE RANGES (in MIDI note numbers)
# ============================================================================
VOICE_RANGES = {
    'T1': (60, 79),   # C4 to G5 (Tenor 1 - melody)
    'T2': (55, 74),   # G3 to D5 (Tenor 2)
    'Bari': (50, 69), # D3 to A4 (Baritone)
    'Bass': (40, 62)  # E2 to D4 (Bass)
}

# Common chord types in Roman numeral analysis
CHORD_TONES = {
    'I': [0, 4, 7],      # Major triad
    'ii': [2, 5, 9],     # Minor triad
    'iii': [4, 7, 11],   # Minor triad
    'IV': [5, 9, 0],     # Major triad
    'V': [7, 11, 2],     # Major triad
    'vi': [9, 0, 4],     # Minor triad
    'vii°': [11, 2, 5],  # Diminished triad
    'V7': [7, 11, 2, 5]  # Dominant seventh
}


class ChordTransitionModel:
    """
    AI COMPONENT #1: Probabilistic Markov Model for Chord Progressions
    ===================================================================
    This model learns chord transition probabilities from the Bach chorale corpus.
    It applies a data-driven approach to harmonic prediction. The model work to
    answer: "Given the current chord, what chord is most likely to come next?"
    by using a first-order Markov chain trained on real musical data.
    """
    
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = {}
        self.chord_counts = defaultdict(int)
        self.trained = False
        
    def train(self, chorales):
        """Learn chord transition probabilities from a corpus of chorales."""
        print("Training chord transition model on Bach chorales:")
        
        for i, chorale in enumerate(chorales):
            try:
                # Analyze chords in the chorale
                chords = chorale.chordify()
                k = chorale.analyze('key')
                
                prev_chord = None
                for c in chords.recurse().getElementsByClass('Chord'):
                    # Get Roman numeral analysis
                    try:
                        rn = music21.roman.romanNumeralFromChord(c, k)
                        current_chord = rn.figure
                        
                        # Clean up the chord symbol
                        current_chord = self._simplify_chord(current_chord)
                        
                        if prev_chord is not None:
                            self.transition_counts[prev_chord][current_chord] += 1
                        
                        self.chord_counts[current_chord] += 1
                        prev_chord = current_chord
                        
                    except:
                        continue
                        
            except Exception as e:
                continue
        
        # Convert counts to probabilities
        self._compute_probabilities()
        self.trained = True
        print(f"Model trained on {len(self.chord_counts)} unique chord types")
        
    def _simplify_chord(self, chord_symbol):
        """Simplify chord symbols to common types."""
        # Map complex symbols to simpler ones
        simplified = chord_symbol.replace('[', '').replace(']', '')
        
        # Keep only basic chord types
        basic_chords = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii', 'i', 'II', 'III', 'VI', 'VII']
        
        for bc in basic_chords:
            if simplified.startswith(bc):
                if '7' in simplified:
                    return bc + '7'
                if 'o' in simplified or '°' in simplified:
                    return bc + '°'
                return bc
        
        return 'I'  # Default
    
    def _compute_probabilities(self):
        """Convert transition counts to probabilities."""
        for prev_chord, next_chords in self.transition_counts.items():
            total = sum(next_chords.values())
            self.transition_probs[prev_chord] = {
                nc: count / total 
                for nc, count in next_chords.items()
            }
    
    def predict_next_chord(self, current_chord, temperature=1.0):
        """
        Predict the next chord using learned probabilities.
        
        Temperature controls randomness:
        - temperature=0: Always pick most likely
        - temperature=1: Sample according to learned distribution
        - temperature>1: More random exploration
        """
        current_chord = self._simplify_chord(current_chord)
        
        if current_chord not in self.transition_probs:
            # Fallback to common progressions
            common_next = {'I': 'IV', 'IV': 'V', 'V': 'I', 'ii': 'V', 'vi': 'ii'}
            return common_next.get(current_chord, 'I')
        
        probs = self.transition_probs[current_chord]
        
        if temperature == 0:
            return max(probs, key=probs.get)
        
        # Sample with temperature
        chords = list(probs.keys())
        weights = [p ** (1/temperature) for p in probs.values()]
        total = sum(weights)
        weights = [w/total for w in weights]
        
        return random.choices(chords, weights=weights)[0]
    
    def get_top_predictions(self, current_chord, n=3):
        """Get top N most likely next chords."""
        current_chord = self._simplify_chord(current_chord)
        
        if current_chord not in self.transition_probs:
            return ['I', 'IV', 'V'][:n]
        
        probs = self.transition_probs[current_chord]
        sorted_chords = sorted(probs.items(), key=lambda x: -x[1])
        return [c for c, p in sorted_chords[:n]]
    
    def save(self, filepath):
        """Save the trained model."""
        data = {
            'transition_probs': dict(self.transition_probs),
            'chord_counts': dict(self.chord_counts)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath):
        """Load a pre-trained model."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.transition_probs = data['transition_probs']
        self.chord_counts = defaultdict(int, data['chord_counts'])
        self.trained = True


class VoiceLeadingConstraints:
    """
    AI COMPONENT #2: Constraint Satisfaction System
    ================================================
    This implements hard and soft constraints for voice leading,
    applying the AI constraint satisfaction problem (CSP).
    
    Hard constraints: Must be satisfied (e.g., voice ranges)
    Soft constraints: Should be satisfied, scored by penalty (e.g., parallel fifths)
    """
    
    @staticmethod
    def check_range(pitch_midi, voice):
        """Hard constraint: pitch must be within voice range."""
        low, high = VOICE_RANGES[voice]
        return low <= pitch_midi <= high
    
    @staticmethod
    def parallel_fifths(prev_voices, curr_voices):
        """
        Soft constraint: Detect parallel fifths (bad voice leading).
        Returns penalty score (0 = no violation, higher = worse).
        """
        penalty = 0
        voices = ['T1', 'T2', 'Bari', 'Bass']
        
        for i in range(len(voices)):
            for j in range(i+1, len(voices)):
                if prev_voices[i] is None or prev_voices[j] is None:
                    continue
                if curr_voices[i] is None or curr_voices[j] is None:
                    continue
                    
                prev_interval = abs(prev_voices[i] - prev_voices[j]) % 12
                curr_interval = abs(curr_voices[i] - curr_voices[j]) % 12
                
                # Check for parallel perfect fifths (7 semitones)
                if prev_interval == 7 and curr_interval == 7:
                    # Check if both voices moved in same direction
                    if (curr_voices[i] - prev_voices[i]) * (curr_voices[j] - prev_voices[j]) > 0:
                        penalty += 10
        
        return penalty
    
    @staticmethod
    def parallel_octaves(prev_voices, curr_voices):
        """Soft constraint: Detect parallel octaves."""
        penalty = 0
        voices = ['T1', 'T2', 'Bari', 'Bass']
        
        for i in range(len(voices)):
            for j in range(i+1, len(voices)):
                if prev_voices[i] is None or prev_voices[j] is None:
                    continue
                if curr_voices[i] is None or curr_voices[j] is None:
                    continue
                    
                prev_interval = abs(prev_voices[i] - prev_voices[j]) % 12
                curr_interval = abs(curr_voices[i] - curr_voices[j]) % 12
                
                if prev_interval == 0 and curr_interval == 0:
                    if (curr_voices[i] - prev_voices[i]) * (curr_voices[j] - prev_voices[j]) > 0:
                        penalty += 10
        
        return penalty
    
    @staticmethod
    def voice_crossing(voices):
        """
        Hard constraint: Voices should not cross.
        T1 >= T2 >= Bari >= Bass
        """
        t1, t2, bari, bass = voices
        if t1 is None or t2 is None or bari is None or bass is None:
            return 0
        
        penalty = 0
        if t2 > t1:
            penalty += 20
        if bari > t2:
            penalty += 20
        if bass > bari:
            penalty += 20
        return penalty
    
    @staticmethod
    def voice_spacing(voices):
        """
        Soft constraint: Prefer reasonable spacing between voices.
        Adjacent voices should generally be within an octave.
        """
        t1, t2, bari, bass = voices
        penalty = 0
        
        pairs = [(t1, t2), (t2, bari), (bari, bass)]
        for upper, lower in pairs:
            if upper is None or lower is None:
                continue
            gap = upper - lower
            if gap > 12:  # More than an octave
                penalty += (gap - 12) * 2
            if gap < 0:  # Voice crossing
                penalty += 20
        
        return penalty
    
    @staticmethod
    def smooth_motion(prev_pitch, curr_pitch):
        """
        Soft constraint: Prefer stepwise motion.
        Large leaps are penalized.
        """
        if prev_pitch is None or curr_pitch is None:
            return 0
        
        leap = abs(curr_pitch - prev_pitch)
        if leap <= 2:  # Step
            return 0
        elif leap <= 4:  # Third
            return 1
        elif leap <= 7:  # Fifth
            return 3
        else:  # Large leap
            return leap - 4


class HarmonizationSearch:
    """
    AI COMPONENT #3: Heuristic Search for Optimal Voicings
    =======================================================
    Given a chord, search for the best voicing that satisfies constraints.
    This uses a search approach with scoring based on constraint violations.
    """
    
    def __init__(self, constraints):
        self.constraints = constraints
    
    def generate_voicings(self, chord_tones, melody_pitch, key_tonic):
        """
        Generate all possible TTBB voicings for a chord.
        
        chord_tones: list of scale degrees (0-11)
        melody_pitch: MIDI pitch of the melody (T1)
        key_tonic: MIDI pitch of the key's tonic
        """
        voicings = []
        
        # Get actual pitches for chord tones in the key
        chord_pitches = [(key_tonic + ct) % 12 for ct in chord_tones]
        
        # T1 is fixed (melody)
        t1 = melody_pitch
        
        # Generate candidates for T2, Bari, Bass
        t2_candidates = self._get_candidates('T2', chord_pitches)
        bari_candidates = self._get_candidates('Bari', chord_pitches)
        bass_candidates = self._get_candidates('Bass', chord_pitches)
        
        # Prefer bass to have root
        root_pitch = chord_pitches[0]
        bass_root_candidates = [b for b in bass_candidates if b % 12 == root_pitch]
        if bass_root_candidates:
            bass_candidates = bass_root_candidates + [b for b in bass_candidates if b % 12 != root_pitch]
        
        # Generate combinations (limit for efficiency)
        for t2 in t2_candidates[:8]:
            for bari in bari_candidates[:8]:
                for bass in bass_candidates[:6]:
                    voicings.append([t1, t2, bari, bass])
        
        return voicings
    
    def _get_candidates(self, voice, chord_pitches):
        """Get all valid pitches for a voice within its range."""
        low, high = VOICE_RANGES[voice]
        candidates = []
        
        for midi_pitch in range(low, high + 1):
            if midi_pitch % 12 in chord_pitches:
                candidates.append(midi_pitch)
        
        return candidates
    
    def score_voicing(self, voicing, prev_voicing=None):
        """
        Score a voicing based on constraint violations.
        Lower score = better voicing.
        """
        score = 0
        
        # Voice crossing
        score += self.constraints.voice_crossing(voicing)
        
        # Voice spacing
        score += self.constraints.voice_spacing(voicing)
        
        # Voice leading from previous chord
        if prev_voicing is not None:
            score += self.constraints.parallel_fifths(prev_voicing, voicing)
            score += self.constraints.parallel_octaves(prev_voicing, voicing)
            
            # Smooth motion for each voice
            for i in range(4):
                score += self.constraints.smooth_motion(prev_voicing[i], voicing[i])
        
        return score
    
    def find_best_voicing(self, chord_tones, melody_pitch, key_tonic, prev_voicing=None, beam_width=10):
        """
        Search to find the best voicing.
        """
        voicings = self.generate_voicings(chord_tones, melody_pitch, key_tonic)
        
        if not voicings:
            return None
        
        # Score all voicings
        scored = [(v, self.score_voicing(v, prev_voicing)) for v in voicings]
        
        # Sort by score (lower is better)
        scored.sort(key=lambda x: x[1])
        
        # Return best voicing
        return scored[0][0] if scored else None


class TTBBHarmonizer:
    """
    Main Harmonization System
    =========================
    Combines all AI components:
    1. Probabilistic chord prediction
    2. Constraint satisfaction
    3. Heuristic search
    """
    
    def __init__(self):
        self.chord_model = ChordTransitionModel()
        self.constraints = VoiceLeadingConstraints()
        self.search = HarmonizationSearch(self.constraints)
        self.trained = False
        
    def train(self, num_chorales=50):
        """Train the system on Bach chorales."""
        print(f"Loading Bach chorales:")
        chorales = list(corpus.chorales.Iterator())[:num_chorales]
        print(f"Loaded {len(chorales)} chorales")
        
        self.chord_model.train(chorales)
        self.trained = True
        
    def harmonize(self, melody_stream, temperature=0.5):
        """
        Create a full TTBB harmonization for a melody.
        
        Args:
            melody_stream: music21 stream containing the melody
            temperature: Controls randomness of chord selection (0=deterministic, 1=random)
        
        Returns:
            music21 Score with 4 parts (T1, T2, Bari, Bass)
        """
        if not self.trained:
            raise ValueError("Model must be trained first!")
        
        # Analyze key
        k = melody_stream.analyze('key')
        key_tonic = k.tonic.midi
        print(f"Detected key: {k}")
        
        # Create output score
        score = stream.Score()
        t1_part = stream.Part()
        t2_part = stream.Part()
        bari_part = stream.Part()
        bass_part = stream.Part()
        
        t1_part.partName = 'Tenor 1'
        t2_part.partName = 'Tenor 2'
        bari_part.partName = 'Baritone'
        bass_part.partName = 'Bass'
        
        # Get melody notes
        melody_notes = list(melody_stream.flatten().notes)
        
        prev_voicing = None
        current_chord = 'I'
        
        for i, mel_note in enumerate(melody_notes):
            # Get melody pitch
            if isinstance(mel_note, note.Note):
                melody_pitch = mel_note.pitch.midi
            else:
                continue
            
            # Determine chord for this beat using AI model
            # Use melody note to influence chord selection
            melody_scale_degree = (melody_pitch - key_tonic) % 12
            
            # Get chord predictions from model
            predicted_chords = self.chord_model.get_top_predictions(current_chord, n=3)
            
            # Filter to chords that contain the melody note
            valid_chords = []
            for pc in predicted_chords:
                if pc in CHORD_TONES:
                    chord_tones = CHORD_TONES[pc]
                    if melody_scale_degree in chord_tones:
                        valid_chords.append(pc)
            
            # If no valid chord, use chord that contains melody
            if not valid_chords:
                for chord_name, tones in CHORD_TONES.items():
                    if melody_scale_degree in tones:
                        valid_chords.append(chord_name)
                        break
            
            if not valid_chords:
                valid_chords = ['I']
            
            # Select chord (with some randomness based on temperature)
            if temperature > 0 and len(valid_chords) > 1:
                selected_chord = random.choice(valid_chords[:2])
            else:
                selected_chord = valid_chords[0]
            
            # Get chord tones
            chord_tones = CHORD_TONES.get(selected_chord, [0, 4, 7])
            
            # Find best voicing using search
            voicing = self.search.find_best_voicing(
                chord_tones, melody_pitch, key_tonic, prev_voicing
            )
            
            if voicing is None:
                # Fallback: simple voicing
                voicing = [melody_pitch, melody_pitch - 4, melody_pitch - 9, melody_pitch - 16]
            
            # Ensure voicing respects ranges (adjust if needed)
            voicing = self._adjust_to_ranges(voicing)
            
            # Create notes for each part
            duration = mel_note.duration
            
            t1_note = note.Note(voicing[0])
            t1_note.duration = duration
            t1_part.append(t1_note)
            
            t2_note = note.Note(voicing[1])
            t2_note.duration = duration
            t2_part.append(t2_note)
            
            bari_note = note.Note(voicing[2])
            bari_note.duration = duration
            bari_part.append(bari_note)
            
            bass_note = note.Note(voicing[3])
            bass_note.duration = duration
            bass_part.append(bass_note)
            
            prev_voicing = voicing
            current_chord = selected_chord
        
        score.append(t1_part)
        score.append(t2_part)
        score.append(bari_part)
        score.append(bass_part)
        
        return score
    
    def _adjust_to_ranges(self, voicing):
        """Adjust voicing to ensure all notes are in valid ranges."""
        voices = ['T1', 'T2', 'Bari', 'Bass']
        adjusted = []
        
        for i, (pitch, voice) in enumerate(zip(voicing, voices)):
            low, high = VOICE_RANGES[voice]
            
            while pitch < low:
                pitch += 12
            while pitch > high:
                pitch -= 12
            
            # Final check
            if pitch < low:
                pitch = low
            if pitch > high:
                pitch = high
            
            adjusted.append(pitch)
        
        return adjusted
    
    def save_model(self, filepath):
        """Save trained model."""
        self.chord_model.save(filepath)
    
    def load_model(self, filepath):
        """Load trained model."""
        self.chord_model.load(filepath)
        self.trained = True


def create_simple_melody():
    """Create a simple test melody."""
    s = stream.Stream()
    s.append(meter.TimeSignature('4/4'))
    s.append(key.Key('C'))
    
    # Simple melody in C major
    pitches = ['C5', 'D5', 'E5', 'D5', 'C5', 'B4', 'C5', 'G4',
               'A4', 'B4', 'C5', 'D5', 'E5', 'D5', 'C5', 'C5']
    
    for p in pitches:
        n = note.Note(p)
        n.duration.type = 'quarter'
        s.append(n)
    
    return s


if __name__ == '__main__':
    # Demo
    print("=" * 60)
    print("TTBB Harmonizer Demo")
    print("=" * 60)
    
    # Create harmonizer
    harmonizer = TTBBHarmonizer()
    
    # Train on Bach chorales
    harmonizer.train(num_chorales=30)
    
    # Create test melody
    melody = create_simple_melody()
    
    # Harmonize
    print("\nGenerating harmonization:")
    result = harmonizer.harmonize(melody, temperature=0.3)
    
    # Save outputs
    print("\nSaving outputs:")
    result.write('midi', 'output_harmonization.mid')
    result.write('musicxml', 'output_harmonization.xml')
    
    print("\nDone! Check output_harmonization.mid and output_harmonization.xml")
