# TBDub inference for FeatherTalk

Use TBDub to generate a speaking avatar video from a silent avatar video and a driving audio track, then upload the result to FeatherTalk.

## Prerequisites

Complete the environment and Student checkpoint setup in the main [README](README.md). Make sure `ffmpeg` and `ffprobe` are available on your `PATH`.

Run the commands below in Bash from the TBDub repository root, with the Python environment activated.

## 1. Prepare the input data

- **Silent avatar video:** prepare a two-minute video of the avatar that loops seamlessly. The end must transition naturally back to the beginning.
- **Driving audio:** prepare a two-minute audio track with pauses between sentences or paragraphs. These pauses help produce better results for silent inference.

The `data/` directory contains example inputs:

- [`driving_video_mousa4.mp4`](data/driving_video_mousa4.mp4): silent avatar video.
- [`driving_audio.wav`](data/driving_audio.wav): recommended default driving audio, with an ideal balance of speech and pauses for this workflow.

You can use the default audio with your own silent avatar video. If you use a different audio track, retain pauses between sentences or paragraphs instead of using continuous speech. Use the default track as a reference for the balance of speech and silence.

## 2. Run inference

Pass the silent avatar video and driving audio to the script:

```bash
./scripts/tbdub.sh <silent_avatar_video_2m> <driving_audio_with_pauses_2m>
```

For example, using the supplied files:

```bash
./scripts/tbdub.sh data/driving_video_mousa4.mp4 data/driving_audio.wav
```

Wait for processing to finish. By default, the script saves the generated video as `tbdub_final.mp4` in the repository root. You can optionally provide an output path as the third argument:

```bash
./scripts/tbdub.sh data/driving_video_mousa4.mp4 data/driving_audio.wav avatar_speaking.mp4
```

## 3. Upload to FeatherTalk

Once inference finishes, upload both of the following to FeatherTalk for the same avatar:

1. The generated speaking video (`tbdub_final.mp4`, or your custom output file).
2. The original silent driving video used as the input to TBDub.
