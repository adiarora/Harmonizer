"""
===============================================
Main Runner Script for TTBB Harmonizer Project
===============================================
"""

import os
from music21 import stream, note, key, meter, corpus
from harmonizer import TTBBHarmonizer, create_simple_melody
from evaluation import HarmonizationEvaluator, compare_with_baseline, generate_baseline_harmonization
import json

# Create output directory
os.makedirs('outputs', exist_ok=True)


def create_folk_melody():
    """Create a folk-style melody (Twinkle Twinkle pattern)."""
    s = stream.Stream()
    s.append(meter.TimeSignature('4/4'))
    s.append(key.Key('G'))
    
    # Twinkle Twinkle style in G major
    pitches = ['G4', 'G4', 'D5', 'D5', 'E5', 'E5', 'D5',
               'C5', 'C5', 'B4', 'B4', 'A4', 'A4', 'G4']
    durations = [1, 1, 1, 1, 1, 1, 2,
                 1, 1, 1, 1, 1, 1, 2]
    
    for p, d in zip(pitches, durations):
        n = note.Note(p)
        n.duration.quarterLength = d
        s.append(n)
    
    return s


def create_hymn_melody():
    """Create a hymn-style melody."""
    s = stream.Stream()
    s.append(meter.TimeSignature('4/4'))
    s.append(key.Key('F'))
    
    # Hymn-like progression in F major
    pitches = ['F4', 'A4', 'C5', 'A4', 'G4', 'F4', 'E4', 'F4',
               'G4', 'A4', 'Bb4', 'A4', 'G4', 'F4', 'E4', 'F4']
    
    for p in pitches:
        n = note.Note(p)
        n.duration.quarterLength = 1
        s.append(n)
    
    return s


def create_ascending_melody():
    """Create an ascending scale melody for testing."""
    s = stream.Stream()
    s.append(meter.TimeSignature('4/4'))
    s.append(key.Key('C'))
    
    pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
               'C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4']
    
    for p in pitches:
        n = note.Note(p)
        n.duration.quarterLength = 1
        s.append(n)
    
    return s


def run_full_evaluation():
    """Run evaluation on different melodies."""
    print("=" * 70)
    print("TTBB HARMONIZER - FULL EVALUATION")
    print("=" * 70)
    
    # Initialize harmonizer
    print("\n[1/5] Training harmonizer on Bach chorales:")
    harmonizer = TTBBHarmonizer()
    harmonizer.train(num_chorales=40)
    
    # Save trained model
    harmonizer.save_model('outputs/chord_model.json')
    print("Model saved to outputs/chord_model.json")
    
    # Test melodies
    test_melodies = {
        'simple_c_major': create_simple_melody(),
        'folk_g_major': create_folk_melody(),
        'hymn_f_major': create_hymn_melody(),
        'scale_c_major': create_ascending_melody()
    }
    
    all_results = {}
    
    print("\n[2/5] Creating harmonizations for test melodies:")
    for name, melody in test_melodies.items():
        print(f"\n  Processing: {name}")
        
        # Generate harmonization
        result = harmonizer.harmonize(melody, temperature=0.3)
        
        # Save outputs
        result.write('midi', f'outputs/{name}_harmonized.mid')
        result.write('musicxml', f'outputs/{name}_harmonized.xml')
        
        # Evaluate
        evaluator = HarmonizationEvaluator()
        eval_results = evaluator.evaluate(result)
        all_results[name] = eval_results
        
        print(f"    - Consonance rate: {eval_results['summary']['consonance_rate']*100:.1f}%")
        print(f"    - Smoothness score: {eval_results['summary']['avg_smoothness_score']:.1f}/10")
        print(f"    - Range violations: {eval_results['range_violations']['total']}")
    
    # Save all evaluation results
    with open('outputs/all_evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n[3/5] Running baseline comparison:")
    # Detailed comparison for one melody
    comparison = compare_with_baseline(harmonizer, test_melodies['simple_c_major'], HarmonizationEvaluator())
    
    with open('outputs/baseline_comparison.json', 'w') as f:
        json.dump({
            'ai_summary': comparison['ai_evaluation']['summary'],
            'baseline_summary': comparison['baseline_evaluation']['summary']
        }, f, indent=2, default=str)
    
    print("\n[4/5] Generating summary statistics:")
    
    # Aggregate statistics across all melodies
    aggregate = {
        'avg_consonance': sum(r['summary']['consonance_rate'] for r in all_results.values()) / len(all_results),
        'avg_smoothness': sum(r['summary']['avg_smoothness_score'] for r in all_results.values()) / len(all_results),
        'total_range_violations': sum(r['range_violations']['total'] for r in all_results.values()),
        'total_voice_crossings': sum(r['voice_crossing']['count'] for r in all_results.values()),
        'total_parallel_fifths': sum(r['parallel_fifths']['count'] for r in all_results.values()),
    }
    
    print("\n" + "-" * 50)
    print("AGGREGATE RESULTS ACROSS ALL TEST MELODIES")
    print("-" * 50)
    print(f"Average consonance rate: {aggregate['avg_consonance']*100:.1f}%")
    print(f"Average smoothness score: {aggregate['avg_smoothness']:.1f}/10")
    print(f"Total range violations: {aggregate['total_range_violations']}")
    print(f"Total voice crossings: {aggregate['total_voice_crossings']}")
    print(f"Total parallel fifths: {aggregate['total_parallel_fifths']}")
    
    with open('outputs/aggregate_stats.json', 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    print("\n[5/5] Generating output file list:")
    
    output_files = os.listdir('outputs')
    print("\nGenerated files:")
    for f in sorted(output_files):
        size = os.path.getsize(f'outputs/{f}')
        print(f"  - {f}")
    
    print("\n" + "=" * 70)
    print("FINISHED EVALUATION")
    print("=" * 70)
    print("\nAll outputs saved to the 'outputs/' directory.")
    print("You can open .mid files in any MIDI player to hear the harmonizations.")
    print("You can open .xml files in MuseScore or Finale to see the sheet music.")
    
    return all_results, aggregate


if __name__ == '__main__':
    results, aggregate = run_full_evaluation()
