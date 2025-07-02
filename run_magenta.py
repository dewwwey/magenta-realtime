import os
import sys
import numpy as np
import soundfile as sf
import time
import gc
import jax

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))

from magenta_rt import system
from magenta_rt import audio

def main():
    total_start_time = time.time()

    print("Initializing MagentaRT...")
    init_start_time = time.time()
    mrt = system.MagentaRT(tag="base", skip_cache=True, lazy=False, device="gpu") # Use GPU, lazy=False for warm_start
    init_end_time = time.time()
    print(f"MagentaRT initialization took {init_end_time - init_start_time:.2f} seconds.")

    print("Attempting to embed style...")
    embed_start_time = time.time()
    try:
        style_embedding = mrt.embed_style("ambient electronic music")
        print("Style embedding successful.")
        jax.clear_caches()
        gc.collect()
    except Exception as e:
        print(f"Error embedding style: {e}")
        return
    embed_end_time = time.time()
    print(f"Style embedding took {embed_end_time - embed_start_time:.2f} seconds.")

    print("Attempting to generate 5 seconds of audio...")
    num_seconds = 5 # Reduced target length
    num_chunks = round(num_seconds / mrt.config.chunk_length)
    all_chunks = []
    state = None

    generation_start_time = time.time()

    for i in range(num_chunks):
        try:
            chunk_waveform, state = mrt.generate_chunk(
                state=state,
                style=style_embedding,
                seed=0, # Use a fixed seed for reproducibility
                temperature=1.0,
                topk=40,
                guidance_weight=5.0
            )
            all_chunks.append(chunk_waveform)
            jax.clear_caches()
            gc.collect()
        except Exception as e:
            print(f"Error generating audio chunk {i+1}: {e}")
            return

    generation_end_time = time.time()
    print(f"Audio generation loop took {generation_end_time - generation_start_time:.2f} seconds.")

    print("Concatenating audio chunks...")
    concat_start_time = time.time()
    final_waveform = audio.concatenate(all_chunks, crossfade_time=mrt.config.crossfade_length)
    concat_end_time = time.time()
    print(f"Audio concatenation took {concat_end_time - concat_start_time:.2f} seconds.")

    output_path = "generated_audio.wav"
    sf.write(output_path, final_waveform.samples, final_waveform.sample_rate)
    print(f"Generated audio saved to {output_path}")

    total_end_time = time.time()
    print(f"Total script execution time: {total_end_time - total_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()