
import os
import sys
import numpy as np
import soundfile as sf
import pynvml

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))

def log_vram_usage(msg=""):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"VRAM Usage {msg}: {info.used / 1024**2:.2f} MB / {info.total / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Failed to log VRAM usage: {e}")

from magenta_rt import system
from magenta_rt import audio

def main():
    log_vram_usage("(Initial)")
    print("Initializing MagentaRT...")
    mrt = system.MagentaRT(tag="base", skip_cache=True, lazy=True, device="gpu")
    log_vram_usage("(After Init)")

    print("Embedding base styles...")
    try:
        style_jazz = mrt.embed_style("jazz")
        style_funk = mrt.embed_style("funk")
        style_electronic = mrt.embed_style("electronic")
        print("Style embeddings created.")
    except Exception as e:
        print(f"Error embedding styles: {e}")
        return

    log_vram_usage("(After Embedding)")

    num_chunks = 10
    all_chunks = []
    state = None

    print(f"Generating {num_chunks} audio chunks with dynamic style blending...")
    for i in range(num_chunks):
        # Blend styles: 0-4 (jazz->funk), 5-9 (funk->electronic)
        if i < 5:
            # Transition from jazz to funk
            funk_weight = i / 4.0
            jazz_weight = 1.0 - funk_weight
            electronic_weight = 0.0
            print(f"Chunk {i+1}/{num_chunks}: Blending Jazz ({jazz_weight:.2f}) and Funk ({funk_weight:.2f})")
            blended_style = (jazz_weight * style_jazz) + (funk_weight * style_funk)
        else:
            # Transition from funk to electronic
            electronic_weight = (i - 5) / 4.0
            funk_weight = 1.0 - electronic_weight
            jazz_weight = 0.0
            print(f"Chunk {i+1}/{num_chunks}: Blending Funk ({funk_weight:.2f}) and Electronic ({electronic_weight:.2f})")
            blended_style = (funk_weight * style_funk) + (electronic_weight * style_electronic)

        try:
            chunk_waveform, state = mrt.generate_chunk(
                state=state,
                style=blended_style,
                seed=42, # Use a fixed seed for reproducibility
            )
            all_chunks.append(chunk_waveform)
        except Exception as e:
            print(f"Error generating audio chunk {i+1}: {e}")
            log_vram_usage(f"(Error on chunk {i+1})")
            return

    print("Audio generation complete.")
    log_vram_usage("(After Generation)")

    print("Concatenating audio chunks...")
    final_waveform = audio.concatenate(all_chunks, crossfade_time=mrt.config.crossfade_length)

    output_path = "dynamic_style_audio.wav"
    sf.write(output_path, final_waveform.samples, final_waveform.sample_rate)
    print(f"Generated audio saved to {output_path}")
    log_vram_usage("(Final)")


if __name__ == "__main__":
    main()
