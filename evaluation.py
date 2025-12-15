"""
Evaluation Module for TTBB Harmonizer
=====================================
Provides quantitative metrics to evaluate harmonization quality.
These metrics are essential for the project report.
"""

from music21 import stream, note, interval, analysis
from collections import defaultdict
import json

# Voice ranges for checking
VOICE_RANGES = {
    'T1': (60, 79),
    'T2': (55, 74),
    'Bari': (50, 69),
    'Bass': (40, 62)
}


class HarmonizationEvaluator:
    """
    Evaluates harmonization quality using multiple metrics.
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, score):
        """
        Run all evaluations on a harmonized score.
        
        Args:
            score: music21 Score with 4 parts
        
        Returns:
            Dictionary of evaluation metrics
        """
        parts = list(score.parts)
        if len(parts) != 4:
            raise ValueError(f"Expected 4 parts, got {len(parts)}")
        
        t1 = list(parts[0].flatten().notes)
        t2 = list(parts[1].flatten().notes)
        bari = list(parts[2].flatten().notes)
        bass = list(parts[3].flatten().notes)
        
        results = {
            'range_violations': self._check_ranges(t1, t2, bari, bass),
            'voice_crossing': self._check_voice_crossing(t1, t2, bari, bass),
            'parallel_fifths': self._check_parallel_fifths(t1, t2, bari, bass),
            'parallel_octaves': self._check_parallel_octaves(t1, t2, bari, bass),
            'consonance_analysis': self._analyze_consonance(t1, t2, bari, bass),
            'voice_leading_smoothness': self._analyze_smoothness(t1, t2, bari, bass),
            'chord_completeness': self._analyze_chord_completeness(t1, t2, bari, bass),
            'summary': {}
        }
        
        # Calculate summary statistics
        total_notes = len(t1)
        results['summary'] = {
            'total_notes': total_notes,
            'range_violation_rate': results['range_violations']['total'] / (total_notes * 4) if total_notes > 0 else 0,
            'voice_crossing_rate': results['voice_crossing']['count'] / total_notes if total_notes > 0 else 0,
            'parallel_fifth_rate': results['parallel_fifths']['count'] / max(total_notes - 1, 1),
            'parallel_octave_rate': results['parallel_octaves']['count'] / max(total_notes - 1, 1),
            'consonance_rate': results['consonance_analysis']['consonance_rate'],
            'avg_smoothness_score': results['voice_leading_smoothness']['avg_score'],
        }
        
        self.results = results
        return results
    
    def _check_ranges(self, t1, t2, bari, bass):
        """Check for range violations in each voice."""
        violations = {'T1': 0, 'T2': 0, 'Bari': 0, 'Bass': 0, 'total': 0}
        
        voice_parts = [('T1', t1), ('T2', t2), ('Bari', bari), ('Bass', bass)]
        
        for voice_name, notes in voice_parts:
            low, high = VOICE_RANGES[voice_name]
            for n in notes:
                if isinstance(n, note.Note):
                    midi = n.pitch.midi
                    if midi < low or midi > high:
                        violations[voice_name] += 1
                        violations['total'] += 1
        
        return violations
    
    def _check_voice_crossing(self, t1, t2, bari, bass):
        """Check for voice crossing errors."""
        crossings = {'count': 0, 'locations': []}
        
        for i in range(len(t1)):
            try:
                pitches = [
                    t1[i].pitch.midi if isinstance(t1[i], note.Note) else None,
                    t2[i].pitch.midi if isinstance(t2[i], note.Note) else None,
                    bari[i].pitch.midi if isinstance(bari[i], note.Note) else None,
                    bass[i].pitch.midi if isinstance(bass[i], note.Note) else None
                ]
                
                if None in pitches:
                    continue
                
                # Check ordering: T1 >= T2 >= Bari >= Bass
                if pitches[1] > pitches[0] or pitches[2] > pitches[1] or pitches[3] > pitches[2]:
                    crossings['count'] += 1
                    crossings['locations'].append(i)
            except:
                continue
        
        return crossings
    
    def _check_parallel_fifths(self, t1, t2, bari, bass):
        """Detect parallel perfect fifths between voice pairs."""
        parallels = {'count': 0, 'locations': []}
        voices = [t1, t2, bari, bass]
        voice_names = ['T1', 'T2', 'Bari', 'Bass']
        
        for i in range(len(t1) - 1):
            for v1 in range(4):
                for v2 in range(v1 + 1, 4):
                    try:
                        prev1 = voices[v1][i].pitch.midi
                        prev2 = voices[v2][i].pitch.midi
                        curr1 = voices[v1][i+1].pitch.midi
                        curr2 = voices[v2][i+1].pitch.midi
                        
                        prev_interval = abs(prev1 - prev2) % 12
                        curr_interval = abs(curr1 - curr2) % 12
                        
                        # Perfect fifth = 7 semitones
                        if prev_interval == 7 and curr_interval == 7:
                            # Check parallel motion
                            motion1 = curr1 - prev1
                            motion2 = curr2 - prev2
                            if motion1 != 0 and motion2 != 0 and motion1 * motion2 > 0:
                                parallels['count'] += 1
                                parallels['locations'].append({
                                    'position': i,
                                    'voices': (voice_names[v1], voice_names[v2])
                                })
                    except:
                        continue
        
        return parallels
    
    def _check_parallel_octaves(self, t1, t2, bari, bass):
        """Detect parallel octaves/unisons between voice pairs."""
        parallels = {'count': 0, 'locations': []}
        voices = [t1, t2, bari, bass]
        voice_names = ['T1', 'T2', 'Bari', 'Bass']
        
        for i in range(len(t1) - 1):
            for v1 in range(4):
                for v2 in range(v1 + 1, 4):
                    try:
                        prev1 = voices[v1][i].pitch.midi
                        prev2 = voices[v2][i].pitch.midi
                        curr1 = voices[v1][i+1].pitch.midi
                        curr2 = voices[v2][i+1].pitch.midi
                        
                        prev_interval = abs(prev1 - prev2) % 12
                        curr_interval = abs(curr1 - curr2) % 12
                        
                        # Octave/unison = 0 semitones
                        if prev_interval == 0 and curr_interval == 0:
                            motion1 = curr1 - prev1
                            motion2 = curr2 - prev2
                            if motion1 != 0 and motion2 != 0 and motion1 * motion2 > 0:
                                parallels['count'] += 1
                                parallels['locations'].append({
                                    'position': i,
                                    'voices': (voice_names[v1], voice_names[v2])
                                })
                    except:
                        continue
        
        return parallels
    
    def _analyze_consonance(self, t1, t2, bari, bass):
        """Analyze consonance/dissonance of vertical intervals."""
        consonant_intervals = {0, 3, 4, 5, 7, 8, 9}  # P1, m3, M3, P4, P5, m6, M6
        
        total_intervals = 0
        consonant_count = 0
        interval_distribution = defaultdict(int)
        
        for i in range(len(t1)):
            try:
                pitches = [
                    t1[i].pitch.midi,
                    t2[i].pitch.midi,
                    bari[i].pitch.midi,
                    bass[i].pitch.midi
                ]
                
                # Check all pairs
                for v1 in range(4):
                    for v2 in range(v1 + 1, 4):
                        intv = abs(pitches[v1] - pitches[v2]) % 12
                        interval_distribution[intv] += 1
                        total_intervals += 1
                        if intv in consonant_intervals:
                            consonant_count += 1
            except:
                continue
        
        return {
            'consonance_rate': consonant_count / total_intervals if total_intervals > 0 else 0,
            'consonant_intervals': consonant_count,
            'total_intervals': total_intervals,
            'interval_distribution': dict(interval_distribution)
        }
    
    def _analyze_smoothness(self, t1, t2, bari, bass):
        """Analyze voice leading smoothness (prefer small intervals)."""
        voices = [t1, t2, bari, bass]
        total_motion = 0
        motion_count = 0
        leap_count = 0  # Intervals > 4 semitones
        
        for voice in voices:
            for i in range(len(voice) - 1):
                try:
                    prev = voice[i].pitch.midi
                    curr = voice[i+1].pitch.midi
                    motion = abs(curr - prev)
                    total_motion += motion
                    motion_count += 1
                    if motion > 4:
                        leap_count += 1
                except:
                    continue
        
        avg_motion = total_motion / motion_count if motion_count > 0 else 0
        
        # Score: 10 = perfect (all stepwise), 0 = bad (all large leaps)
        smoothness_score = max(0, 10 - avg_motion)
        
        return {
            'avg_motion': avg_motion,
            'avg_score': smoothness_score,
            'total_leaps': leap_count,
            'leap_rate': leap_count / motion_count if motion_count > 0 else 0
        }
    
    def _analyze_chord_completeness(self, t1, t2, bari, bass):
        """Check if chords generally have root, third, and fifth."""
        complete_count = 0
        total_count = 0
        
        for i in range(len(t1)):
            try:
                pitches = {
                    t1[i].pitch.midi % 12,
                    t2[i].pitch.midi % 12,
                    bari[i].pitch.midi % 12,
                    bass[i].pitch.midi % 12
                }
                
                # A complete triad has 3 distinct pitch classes
                # (assuming some doubling is okay)
                total_count += 1
                if len(pitches) >= 3:
                    complete_count += 1
            except:
                continue
        
        return {
            'completeness_rate': complete_count / total_count if total_count > 0 else 0,
            'complete_chords': complete_count,
            'total_chords': total_count
        }
    
    def print_report(self):
        """Print a formatted evaluation report."""
        if not self.results:
            print("No evaluation results. Run evaluate() first.")
            return
        
        r = self.results
        s = r['summary']
        
        print("\n" + "=" * 60)
        print("HARMONIZATION EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\nTotal notes analyzed: {s['total_notes']}")
        
        print("\n--- Range Compliance ---")
        print(f"Range violations: {r['range_violations']['total']} ({s['range_violation_rate']*100:.1f}%)")
        for voice in ['T1', 'T2', 'Bari', 'Bass']:
            print(f"  {voice}: {r['range_violations'][voice]} violations")
        
        print("\n--- Voice Leading Quality ---")
        print(f"Voice crossings: {r['voice_crossing']['count']} ({s['voice_crossing_rate']*100:.1f}%)")
        print(f"Parallel fifths: {r['parallel_fifths']['count']} ({s['parallel_fifth_rate']*100:.1f}%)")
        print(f"Parallel octaves: {r['parallel_octaves']['count']} ({s['parallel_octave_rate']*100:.1f}%)")
        
        print("\n--- Harmonic Quality ---")
        print(f"Consonance rate: {s['consonance_rate']*100:.1f}%")
        print(f"Voice leading smoothness: {s['avg_smoothness_score']:.1f}/10")
        
        print("\n--- Chord Quality ---")
        print(f"Chord completeness: {r['chord_completeness']['completeness_rate']*100:.1f}%")
        
        print("\n" + "=" * 60)
        
        # Overall assessment
        score = 0
        if s['range_violation_rate'] < 0.05:
            score += 2
        if s['voice_crossing_rate'] < 0.1:
            score += 2
        if s['consonance_rate'] > 0.8:
            score += 2
        if s['avg_smoothness_score'] > 6:
            score += 2
        if s['parallel_fifth_rate'] < 0.1:
            score += 1
        if s['parallel_octave_rate'] < 0.1:
            score += 1
        
        print(f"OVERALL QUALITY SCORE: {score}/10")
        
        if score >= 8:
            print("Assessment: Excellent harmonization")
        elif score >= 6:
            print("Assessment: Good harmonization with minor issues")
        elif score >= 4:
            print("Assessment: Acceptable but needs improvement")
        else:
            print("Assessment: Significant issues detected")
        
        print("=" * 60)
    
    def save_results(self, filepath):
        """Save results to JSON file."""
        # Convert defaultdict to regular dict for JSON serialization
        results_copy = json.loads(json.dumps(self.results, default=str))
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)


def compare_with_baseline(harmonizer, melody, evaluator):
    """
    Compare learned harmonizer with a simple baseline.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: AI Model vs Simple Baseline")
    print("=" * 60)
    
    # Generate harmonization with AI
    print("\n1. Generating AI harmonization:")
    ai_result = harmonizer.harmonize(melody, temperature=0.3)
    ai_eval = evaluator.evaluate(ai_result)
    
    # Generate simple baseline (just stack thirds)
    print("\n2. Generating baseline harmonization:")
    baseline_result = generate_baseline_harmonization(melody)
    baseline_evaluator = HarmonizationEvaluator()
    baseline_eval = baseline_evaluator.evaluate(baseline_result)
    
    # Compare
    print("\n" + "-" * 40)
    print("COMPARISON RESULTS")
    print("-" * 40)
    
    metrics = [
        ('Range Violation Rate', 'range_violation_rate', False),
        ('Voice Crossing Rate', 'voice_crossing_rate', False),
        ('Consonance Rate', 'consonance_rate', True),
        ('Smoothness Score', 'avg_smoothness_score', True),
    ]
    
    print(f"\n{'Metric':<25} {'AI Model':<15} {'Baseline':<15} {'Winner':<10}")
    print("-" * 65)
    
    ai_wins = 0
    for name, key, higher_better in metrics:
        ai_val = ai_eval['summary'][key]
        base_val = baseline_eval['summary'][key]
        
        if higher_better:
            winner = "AI" if ai_val > base_val else "Baseline" if base_val > ai_val else "Tie"
        else:
            winner = "AI" if ai_val < base_val else "Baseline" if base_val < ai_val else "Tie"
        
        if winner == "AI":
            ai_wins += 1
        
        # Format values
        if 'rate' in key.lower():
            ai_str = f"{ai_val*100:.1f}%"
            base_str = f"{base_val*100:.1f}%"
        else:
            ai_str = f"{ai_val:.2f}"
            base_str = f"{base_val:.2f}"
        
        print(f"{name:<25} {ai_str:<15} {base_str:<15} {winner:<10}")
    
    print("-" * 65)
    print(f"\nAI Model wins on {ai_wins}/{len(metrics)} metrics")
    
    return {
        'ai_evaluation': ai_eval,
        'baseline_evaluation': baseline_eval
    }


def generate_baseline_harmonization(melody_stream):
    """
    Simple baseline: harmonize by stacking fixed intervals below melody.
    This doesn't use any learning or sophisticated rules.
    """
    from music21 import stream, note, key, meter
    
    score = stream.Score()
    t1_part = stream.Part()
    t2_part = stream.Part()
    bari_part = stream.Part()
    bass_part = stream.Part()
    
    melody_notes = list(melody_stream.flatten().notes)
    
    for mel_note in melody_notes:
        if isinstance(mel_note, note.Note):
            melody_pitch = mel_note.pitch.midi
            duration = mel_note.duration
            
            # Simple stacking: T2 = melody - 4 (third), Bari = melody - 7 (fifth), Bass = melody - 12 (octave)
            t1 = note.Note(melody_pitch)
            t1.duration = duration
            t1_part.append(t1)
            
            t2 = note.Note(max(55, melody_pitch - 4))  # Clamp to range
            t2.duration = duration
            t2_part.append(t2)
            
            bari = note.Note(max(50, melody_pitch - 9))
            bari.duration = duration
            bari_part.append(bari)
            
            bass = note.Note(max(40, melody_pitch - 16))
            bass.duration = duration
            bass_part.append(bass)
    
    score.append(t1_part)
    score.append(t2_part)
    score.append(bari_part)
    score.append(bass_part)
    
    return score


if __name__ == '__main__':
    from harmonizer import TTBBHarmonizer, create_simple_melody
    
    # Create and train harmonizer
    harmonizer = TTBBHarmonizer()
    harmonizer.train(num_chorales=30)
    
    # Create melody
    melody = create_simple_melody()
    
    # Generate harmonization
    result = harmonizer.harmonize(melody)
    
    # Evaluate
    evaluator = HarmonizationEvaluator()
    evaluator.evaluate(result)
    evaluator.print_report()
    
    # Save results
    evaluator.save_results('evaluation_results.json')
    
    # Compare with baseline
    compare_with_baseline(harmonizer, melody, evaluator)
