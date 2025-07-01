
import os
import sys
import numpy as np
import soundfile as sf

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))

from magenta_rt import system
from magenta_rt import audio

def main():
    print("Initializing MagentaRT...")
    # Use lazy=True to avoid immediate warm_start which might trigger the XLA error
    # We'll call warm_start explicitly later if needed.
    mrt = system.MagentaRT(tag="base", skip_cache=True, lazy=True, device="gpu") # Use GPU

    print("Attempting to embed style...")
    try:
        style_embedding = mrt.embed_style("ambient electronic music")
        print("Style embedding successful.")
    except Exception as e:
        print(f"Error embedding style: {e}")
        return

    print("Attempting to generate a chunk of audio...")
    try:
        # warm_start is called implicitly by generate_chunk if lazy=True
        audio_waveform, new_state = mrt.generate_chunk(
            state=None,
            style=style_embedding,
            seed=0,
            temperature=1.0,
            topk=40,
            guidance_weight=5.0
        )
        print("Audio chunk generation successful.")

        output_path = "generated_audio.wav"
        sf.write(output_path, audio_waveform.samples, audio_waveform.sample_rate)
        print(f"Generated audio saved to {output_path}")

    except Exception as e:
        print(f"Error generating audio chunk: {e}")
        return

if __name__ == "__main__":
    # Ensure the virtual environment is activated
    # This script assumes it's run after `source .venv/bin/activate`
    main()
