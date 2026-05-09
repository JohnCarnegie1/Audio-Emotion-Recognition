import torch
import torchaudio


def get_audio_transform():
    """
    Returns a transform function for raw audio normalization.
    """

    def transform(waveform):
        """
        waveform shape: [channels, samples]
        """

        # Convert to float
        waveform = waveform.float()

        # Normalize waveform to range [-1, 1]
        waveform = waveform / waveform.abs().max()

        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

        return waveform

    return transform


# Example usage
waveform, sample_rate = torchaudio.load("f_ans001aen.wav")

transform = get_audio_transform()
waveform = transform(waveform)

print(waveform.shape)

