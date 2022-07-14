# ----------------------------
# Convert the given audio to the desired number of channels
# ----------------------------
@staticmethod
def rechannel(aud, new_channel):
    sig, sr = aud
    
    if (sig.shape[0] == new_channel):
        # Nothing to do
        return aud

    if (new_channel == 1):
        # Convert from stereo to mono by selecting only the first channel
        resig = sig[:1, :]
    else:
        # Convert from mono to stereo by duplicating the first channel
        resig = torch.cat([sig, sig])

    eturn ((resig, sr))