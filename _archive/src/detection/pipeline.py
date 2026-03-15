"""
ALL Detection Pipeline Orchestrator
Combines all stages: Screening → Detection → Explanation
"""

import os
import json
import argparse
from datetime import datetime

# Handle both module and standalone imports
try:
    from .stage1_screening import ALLScreener
    from .blast_detector_v5 import detect_blasts
except ImportError:
    from stage1_screening import ALLScreener
    from blast_detector_v5 import detect_blasts


class ALLPipeline:
    """
    Complete ALL detection pipeline.
    
    Stage 1: TFLite screening (is image ALL-positive?)
    Stage 2: V5 detection (which cells are blasts?)
    Stage 3: LLM explanation (why are they blasts?)
    """
    
    def __init__(self, screener_model=None, use_llm=False, llm_model='phi3'):
        """
        Initialize the pipeline.
        
        Args:
            screener_model: Path to TFLite model for Stage 1
            use_llm: Whether to use LLM for explanations
            llm_model: Which Ollama model to use (phi3, gemma3:4b, etc.)
        """
        self.screener = ALLScreener(model_path=screener_model)
        self.use_llm = use_llm
        self.llm_model = llm_model
        
    def analyze(self, image_path, save_json=None):
        """
        Run complete analysis pipeline on an image.
        
        Args:
            image_path: Path to blood smear image
            save_json: Optional path to save results as JSON
            
        Returns:
            dict with complete analysis results
        """
        result = {
            'image': image_path,
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        # --- Stage 1: Screening ---
        print(f"\n{'='*60}")
        print(f"STAGE 1: Image Screening")
        print(f"{'='*60}")
        
        screening = self.screener.predict(image_path)
        result['stages']['screening'] = screening
        
        print(f"Classification: {screening['classification']}")
        print(f"Confidence: {screening['confidence']*100:.1f}%")
        
        if not screening['positive']:
            result['final_result'] = 'HEALTHY'
            result['summary'] = 'No signs of Acute Lymphoblastic Leukemia detected.'
            print(f"\n✓ Image classified as HEALTHY - Skipping further analysis")
            
            if save_json:
                self._save_json(result, save_json)
            return result
        
        # --- Stage 2: Cell Detection ---
        print(f"\n{'='*60}")
        print(f"STAGE 2: Blast Cell Detection")
        print(f"{'='*60}")
        
        detections = detect_blasts(image_path)
        result['stages']['detection'] = detections
        
        print(f"\nTotal cells found: {detections['total_cells']}")
        print(f"Suspected blasts: {detections['blast_count']}")
        
        # --- Stage 3: Explanation ---
        if self.use_llm and detections['blast_count'] > 0:
            print(f"\n{'='*60}")
            print(f"STAGE 3: Generating Explanation (LLM)")
            print(f"{'='*60}")
            
            explanation = self._generate_explanation(detections['detections'])
            result['stages']['explanation'] = explanation
            print(f"\n{explanation}")
        else:
            # Template-based explanation
            explanation = self._template_explanation(detections)
            result['stages']['explanation'] = explanation
        
        # Final result
        if detections['blast_count'] > 0:
            result['final_result'] = 'ALL_POSITIVE'
            result['summary'] = f"ALERT: {detections['blast_count']} suspected blast cells detected."
        else:
            result['final_result'] = 'HEALTHY'
            result['summary'] = 'Screening positive but no blast cells detected by V5.'
        
        if save_json:
            self._save_json(result, save_json)
        
        return result
    
    def _template_explanation(self, detections):
        """Generate template-based explanation (no LLM needed)."""
        if detections['blast_count'] == 0:
            return "No blast cells detected requiring explanation."
        
        blasts = detections['detections']
        avg_area = sum(d['area'] for d in blasts) / len(blasts)
        avg_circ = sum(d['circularity'] for d in blasts) / len(blasts)
        avg_homo = sum(d['homogeneity'] for d in blasts) / len(blasts)
        avg_score = sum(d['score'] for d in blasts) / len(blasts)
        
        return f"""
ANALYSIS SUMMARY
================
Detected {len(blasts)} suspected L1 ALL blast cell(s).

MORPHOLOGICAL FEATURES:
• Average nucleus size: {avg_area:.0f} px² (abnormally large)
• Average roundness: {avg_circ*100:.1f}% (highly regular - characteristic of L1)
• Average chromatin homogeneity: {avg_homo*100:.1f}% (smooth, fine pattern)
• Average blast score: {avg_score:.2f} (threshold: 3.2)

CLINICAL INTERPRETATION:
These cells exhibit morphological features consistent with L1-type Acute 
Lymphoblastic Leukemia blast cells: large round nuclei with smooth chromatin
and minimal cytoplasm (high N:C ratio).

RECOMMENDATION:
Confirmatory testing recommended. This is a screening result only.
"""
    
    def _generate_explanation(self, detections):
        """Generate LLM-based explanation using Ollama."""
        try:
            import ollama
            
            # Extract key metrics for the prompt
            avg_area = sum(d['area'] for d in detections) / len(detections)
            avg_circ = sum(d['circularity'] for d in detections) / len(detections)
            avg_homo = sum(d['homogeneity'] for d in detections) / len(detections)
            avg_score = sum(d['score'] for d in detections) / len(detections)
            
            prompt = f"""DETECTED: {len(detections)} blast cell(s)

MEASUREMENTS:
- Circularity: {avg_circ*100:.0f}%
- Chromatin homogeneity: {avg_homo*100:.0f}%
- Blast score: {avg_score:.2f} (threshold: 3.2)

In exactly 2 sentences, state which measurements exceeded thresholds and why this indicates L1 ALL blasts. Be direct and clinical."""

            response = ollama.chat(model=self.llm_model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
            
        except ImportError:
            return self._template_explanation({'detections': detections, 'blast_count': len(detections)})
        except Exception as e:
            return f"LLM explanation failed: {e}\n" + self._template_explanation({'detections': detections, 'blast_count': len(detections)})
    
    def _save_json(self, result, path):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description='ALL Detection Pipeline')
    parser.add_argument('image', help='Path to blood smear image')
    parser.add_argument('--output', '-o', help='Output JSON path')
    parser.add_argument('--llm', action='store_true', help='Use LLM for explanations')
    parser.add_argument('--model', '-m', help='TFLite screening model path')
    
    args = parser.parse_args()
    
    pipeline = ALLPipeline(
        screener_model=args.model,
        use_llm=args.llm
    )
    
    result = pipeline.analyze(args.image, save_json=args.output)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {result['final_result']}")
    print(f"{'='*60}")
    print(result['summary'])
    
    if 'explanation' in result.get('stages', {}):
        print(result['stages']['explanation'])


if __name__ == "__main__":
    main()
