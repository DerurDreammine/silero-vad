[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb)

![header](https://user-images.githubusercontent.com/12515440/89997349-b3523080-dc94-11ea-9906-ca2e8bc50535.png)

<br/>
<h1 align="center">Silero VAD UPDATED BY DERUR</h1>
<br/>

**Silero VAD** - pre-trained enterprise-grade [Voice Activity Detector](https://en.wikipedia.org/wiki/Voice_activity_detection)
<br/>

<p align="center">
  <img src="https://github.com/snakers4/silero-vad/assets/36505480/300bd062-4da5-4f19-9736-9c144a45d7a7" />
</p>

<br/>

<h2 align="center">Fast start</h2>
<br/>

<details>
<summary>Dependencies</summary>

  System requirements to run python examples on `x86-64` systems:
  
  - `python 3.8+`;
  - 1G+ RAM;
  - A modern CPU with AVX, AVX2, AVX-512 or AMX instruction sets.

  Dependencies:
  
  - `torch>=1.12.0`;
  - `torchaudio>=0.12.0` (for I/O only);
  - `onnxruntime>=1.16.1` (for ONNX model usage).
  
  Silero VAD uses torchaudio library for audio I/O (`torchaudio.info`, `torchaudio.load`, and `torchaudio.save`), so a proper audio backend is required:
  
  - Option №1 - [**FFmpeg**](https://www.ffmpeg.org/) backend. `conda install -c conda-forge 'ffmpeg<7'`;
  - Option №2 - [**sox_io**](https://pypi.org/project/sox/) backend. `apt-get install sox`, TorchAudio is tested on libsox 14.4.2;
  - Option №3 - [**soundfile**](https://pypi.org/project/soundfile/) backend. `pip install soundfile`.

If you are planning to run the VAD using solely the `onnx-runtime`, it will run on any other system architectures where onnx-runtume is [supported](https://onnxruntime.ai/getting-started). In this case please note that:

- You will have to implement the I/O;
- You will have to adapt the existing wrappers / examples / post-processing for your use-case.

</details>

```python3
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
model = load_silero_vad()
wav = read_audio('path_to_audio_file')
speech_timestamps = get_speech_timestamps(
  wav,
  model,
  return_seconds=True,  # Return speech timestamps in seconds (default is samples)
)
print(speech_timestamps)
```

<br/>
