import os
import sys
import numpy as np
import sounddevice as sd
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))

from magenta_rt import system
from magenta_rt import audio

def main():
    print("Initializing MagentaRT...")
    mrt = system.MagentaRT(tag="base", skip_cache=True, lazy=False, device="gpu") # lazy=False for warm_start

    print("Embedding base styles...")
    try:
        style_jazz = mrt.embed_style("jazz")
        style_funk = mrt.embed_style("funk")
        style_electronic = mrt.embed_style("electronic")
        print("Style embeddings created.")
    except Exception as e:
        print(f"Error embedding styles: {e}")
        return

    # Set up audio stream
    samplerate = mrt.sample_rate
    channels = mrt.num_channels
    blocksize = int(mrt.config.chunk_length * samplerate) # Process one chunk at a time

    print("Starting real-time audio stream...")
    try:
        with sd.OutputStream(samplerate=samplerate, channels=channels, blocksize=blocksize) as stream:
            state = None
            chunk_count = 0
            while True:
                # Blend styles: transition every 5 chunks (10 seconds)
                if chunk_count % 10 < 5:
                    # Transition from jazz to funk
                    blend_factor = (chunk_count % 10) / 4.0
                    jazz_weight = 1.0 - blend_factor
                    funk_weight = blend_factor
                    electronic_weight = 0.0
                    print(f"Chunk {chunk_count+1}: Blending Jazz ({jazz_weight:.2f}) and Funk ({funk_weight:.2f})")
                    blended_style = (jazz_weight * style_jazz) + (funk_weight * style_funk)
                else:
                    # Transition from funk to electronic
                    blend_factor = ((chunk_count % 10) - 5) / 4.0
                    funk_weight = 1.0 - blend_factor
                    electronic_weight = blend_factor
                    jazz_weight = 0.0
                    print(f"Chunk {chunk_count+1}: Blending Funk ({funk_weight:.2f}) and Electronic ({electronic_weight:.2f})")
                    blended_style = (funk_weight * style_funk) + (electronic_weight * style_electronic)

                start_gen_time = time.time()
                chunk_waveform, state = mrt.generate_chunk(
                    state=state,
                    style=blended_style,
                    seed=42, # Use a fixed seed for reproducibility
                )
                end_gen_time = time.time()
                gen_duration = end_gen_time - start_gen_time

                # Play the generated chunk
                stream.write(chunk_waveform.samples.astype(np.float32))

                print(f"Generated chunk {chunk_count+1} in {gen_duration:.2f} seconds.")
                chunk_count += 1

    except Exception as e:
        print(f"Error during audio streaming: {e}")

if __name__ == "__main__":
    main()