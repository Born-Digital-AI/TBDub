#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TBDub long-video Student inference - quality/stability version
#
# Main changes vs. the previous Student wrapper:
#   - Student + CPU offload
#   - official recommended Student defaults
#   - context overlap around every external chunk
#   - keeps the ORIGINAL upper face/eyes and uses TBDub mainly for lower face/mouth
#   - frame-accurate 25 fps / 16 kHz chunking
#   - concatenates VIDEO ONLY and muxes clean audio once at the end
#   - validates final frame count and FeatherTalk-compatible audio length
#
# Usage:
#   ./tbdub_long_student_default_limited.sh silent_25fps.mp4 speech.wav output.mp4
#
# Useful overrides:
#   STUDENT_STEPS=2      ./tbdub_long_student_default_limited.sh ...
#   SIGMA_SHIFT=1.0      ./tbdub_long_student_default_limited.sh ...
#   CHUNK_DURATION=10    ./tbdub_long_student_default_limited.sh ...
#   NUM_ITERATIONS=2     ./tbdub_long_student_default_limited.sh ...   # generate only first 2 core chunks
#   CONTEXT_FRAMES=20    ./tbdub_long_student_default_limited.sh ...
#   PRESERVE_UPPER_FACE=0 ./tbdub_long_student_default_limited.sh ...
# ============================================================

VIDEO="${1:-}"
AUDIO="${2:-}"
OUTPUT="${3:-tbdub_final.mp4}"

TARGET_FPS=25
AUDIO_SAMPLE_RATE=16000
SAMPLES_PER_FRAME=$((AUDIO_SAMPLE_RATE / TARGET_FPS))

# Core output chunk. With 1 s context on both sides, 13 s gives a maximum
# 15 s TBDub inference window. You can set CHUNK_DURATION=10 for 10-second tests.
CHUNK_DURATION="${CHUNK_DURATION:-13}"
FRAMES_PER_CHUNK=$((TARGET_FPS * CHUNK_DURATION))

# Context generated but discarded. This moves every kept boundary frame away
# from a fresh Student cold start and lets TBDub build motion context first.
CONTEXT_FRAMES="${CONTEXT_FRAMES:-25}"

# Optional test mode: limit how many core chunks are generated.
# Example: CHUNK_DURATION=10 NUM_ITERATIONS=2 -> generate only the first 20 seconds.
NUM_ITERATIONS="${NUM_ITERATIONS:-${MAX_CHUNKS:-0}}"

# One frame of harmless audio tail keeps FeatherHuBERT from ending short.
FINAL_AUDIO_TAIL_FRAMES=1
FINAL_AUDIO_TAIL_SAMPLES=$((FINAL_AUDIO_TAIL_FRAMES * SAMPLES_PER_FRAME))

# Student quality/speed settings.
# Official TBDub Student quick-start defaults.
# Student does not use Teacher-style reference/audio CFG.
STUDENT_STEPS="${STUDENT_STEPS:-2}"
SIGMA_SHIFT="${SIGMA_SHIFT:-1.0}"
MOTION_FROM_LATENTS="${MOTION_FROM_LATENTS:-1}"
SEED="${SEED:-42}"

# Keep source eyes/upper face. TBDub is used mainly for the mouth/lower face.
# This is especially useful for FeatherTalk training, where we need synthetic
# speech articulation but do not want TBDub to invent blinks/eye texture.
PRESERVE_UPPER_FACE="${PRESERVE_UPPER_FACE:-1}"

# Within TBDub's detected face bbox, alpha starts at 48% of bbox height and
# reaches full generated content at 60%. Eyes remain 100% from source.
LOWER_FACE_BLEND_START="${LOWER_FACE_BLEND_START:-0.48}"
LOWER_FACE_BLEND_FULL="${LOWER_FACE_BLEND_FULL:-0.60}"
LOWER_FACE_X_FADE="${LOWER_FACE_X_FADE:-0.08}"

if [[ -z "$VIDEO" || -z "$AUDIO" ]]; then
    echo "Usage: $0 video.mp4 audio.wav [output.mp4]"
    exit 1
fi

[[ -f "$VIDEO" ]] || { echo "ERROR: Video not found: $VIDEO"; exit 1; }
[[ -f "$AUDIO" ]] || { echo "ERROR: Audio not found: $AUDIO"; exit 1; }
[[ -f "inference.py" ]] || {
    echo "ERROR: inference.py not found. Run this script from the TBDub directory."
    exit 1
}

command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg is not installed."; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "ERROR: ffprobe is not installed."; exit 1; }

get_frame_count() {
    local file="$1"
    local count
    count=$(ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames \
        -of default=noprint_wrappers=1:nokey=1 "$file" | head -n 1)
    if [[ -z "$count" || "$count" == "N/A" ]]; then
        count=$(ffprobe -v error -select_streams v:0 \
            -show_entries stream=nb_frames \
            -of default=noprint_wrappers=1:nokey=1 "$file" | head -n 1)
    fi
    if [[ -z "$count" || "$count" == "N/A" || ! "$count" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Could not determine frame count for: $file" >&2
        return 1
    fi
    echo "$count"
}

get_fps() {
    local file="$1"
    local rate
    rate=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=avg_frame_rate \
        -of default=noprint_wrappers=1:nokey=1 "$file" | head -n 1)
    awk -F/ 'NF == 2 && $2 != 0 { printf "%.6f", $1 / $2; exit } NF == 1 { printf "%.6f", $1; exit }' <<< "$rate"
}

WORKDIR="$(mktemp -d -p . tbdub_long_XXXXXX)"
RESULTS_DIR="$WORKDIR/results"
mkdir -p "$RESULTS_DIR"

cleanup() {
    local status=$?
    echo
    if [[ $status -eq 0 ]]; then
        echo "Cleaning temporary files..."
        rm -rf "$WORKDIR"
    else
        echo "ERROR: Processing failed. Temporary files kept at:"
        echo "  $WORKDIR"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helper: preserve original upper face and blend TBDub only into lower face.
# Uses the exact per-frame face bboxes that TBDub MediaPipe preprocessing wrote
# to --preprocess-report. It writes H.264 directly through ffmpeg stdin.
# ---------------------------------------------------------------------------
BLEND_HELPER="$WORKDIR/blend_lower_face.py"
cat > "$BLEND_HELPER" <<'PY'
import argparse
import json
import subprocess
import sys

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--generated", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--offset", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--blend-start", type=float, default=0.48)
    p.add_argument("--blend-full", type=float, default=0.60)
    p.add_argument("--x-fade", type=float, default=0.08)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not (0.0 <= args.blend_start < args.blend_full <= 1.0):
        raise ValueError("Expected 0 <= blend-start < blend-full <= 1")

    report = json.load(open(args.report, "r", encoding="utf-8"))
    if not report.get("success", False):
        raise RuntimeError(f"TBDub preprocessing report failed: {report.get('error')}")
    bboxes = report.get("bboxes", [])
    if len(bboxes) < args.offset + args.count:
        raise RuntimeError(
            f"Not enough bboxes: have {len(bboxes)}, need {args.offset + args.count}"
        )

    src = cv2.VideoCapture(args.source)
    gen = cv2.VideoCapture(args.generated)
    if not src.isOpened() or not gen.isOpened():
        raise RuntimeError("Could not open source/generated video")

    width = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Skip generated context before the core segment.
    for _ in range(args.offset):
        ok1, _ = src.read()
        ok2, _ = gen.read()
        if not ok1 or not ok2:
            raise RuntimeError("Video ended while skipping context frames")

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        for local_i in range(args.count):
            ok_s, source = src.read()
            ok_g, generated = gen.read()
            if not ok_s or not ok_g:
                raise RuntimeError(f"Video ended early at core frame {local_i}")

            bbox = bboxes[args.offset + local_i]
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            x1 = max(0, min(width - 1, x1))
            x2 = max(x1 + 1, min(width, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(y1 + 1, min(height, y2))

            face_h = y2 - y1
            face_w = x2 - x1
            ys = int(round(face_h * args.blend_start))
            yf = int(round(face_h * args.blend_full))
            yf = max(ys + 1, min(face_h, yf))

            # Vertical alpha: source only above ys, smooth ramp, generated below yf.
            my = np.zeros(face_h, dtype=np.float32)
            my[ys:yf] = np.linspace(0.0, 1.0, yf - ys, endpoint=False, dtype=np.float32)
            my[yf:] = 1.0

            # Horizontal soft edges so lower-face replacement does not form a box.
            mx = np.ones(face_w, dtype=np.float32)
            fade = max(1, int(round(face_w * args.x_fade)))
            fade = min(fade, max(1, face_w // 2))
            mx[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            mx[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)

            alpha = my[:, None] * mx[None, :]
            sigma = max(1.0, min(face_w, face_h) * 0.012)
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
            alpha = np.clip(alpha, 0.0, 1.0)[..., None]

            src_roi = source[y1:y2, x1:x2].astype(np.float32)
            gen_roi = generated[y1:y2, x1:x2].astype(np.float32)
            blended = src_roi * (1.0 - alpha) + gen_roi * alpha
            output = source.copy()
            output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

            proc.stdin.write(output.tobytes())
    finally:
        src.release()
        gen.release()
        if proc.stdin:
            proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder failed with code {rc}")


if __name__ == "__main__":
    main()
PY

SOURCE_FPS=$(get_fps "$VIDEO")
if ! awk -v fps="$SOURCE_FPS" -v target="$TARGET_FPS" \
    'BEGIN { d=fps-target; if (d<0) d=-d; exit !(d < 0.001) }'; then
    echo "ERROR: Input video must be exactly ${TARGET_FPS} fps. Detected: ${SOURCE_FPS}"
    exit 1
fi

TOTAL_FRAMES=$(get_frame_count "$VIDEO")
(( TOTAL_FRAMES > 0 )) || { echo "ERROR: Input video has no frames."; exit 1; }

FULL_NUM_CHUNKS=$(( (TOTAL_FRAMES + FRAMES_PER_CHUNK - 1) / FRAMES_PER_CHUNK ))
if [[ ! "$NUM_ITERATIONS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NUM_ITERATIONS must be a non-negative integer. Got: $NUM_ITERATIONS"
    exit 1
fi
if (( NUM_ITERATIONS > 0 && NUM_ITERATIONS < FULL_NUM_CHUNKS )); then
    NUM_CHUNKS=$NUM_ITERATIONS
else
    NUM_CHUNKS=$FULL_NUM_CHUNKS
fi

OUTPUT_TOTAL_FRAMES=$((NUM_CHUNKS * FRAMES_PER_CHUNK))
(( OUTPUT_TOTAL_FRAMES > TOTAL_FRAMES )) && OUTPUT_TOTAL_FRAMES=$TOTAL_FRAMES

TOTAL_VIDEO_SAMPLES=$((OUTPUT_TOTAL_FRAMES * SAMPLES_PER_FRAME))
MASTER_AUDIO_SAMPLES=$((TOTAL_VIDEO_SAMPLES + FINAL_AUDIO_TAIL_SAMPLES))

VIDEO_DURATION=$(awk -v f="$TOTAL_FRAMES" -v fps="$TARGET_FPS" 'BEGIN { printf "%.6f", f/fps }')
OUTPUT_DURATION=$(awk -v f="$OUTPUT_TOTAL_FRAMES" -v fps="$TARGET_FPS" 'BEGIN { printf "%.6f", f/fps }')
MASTER_AUDIO_DURATION=$(awk -v s="$MASTER_AUDIO_SAMPLES" -v sr="$AUDIO_SAMPLE_RATE" 'BEGIN { printf "%.6f", s/sr }')
MASTER_AUDIO="$WORKDIR/master_audio_16k.wav"

ffmpeg -y -hide_banner -loglevel error \
    -i "$AUDIO" -vn \
    -af "aresample=${AUDIO_SAMPLE_RATE}:async=1:first_pts=0,apad,atrim=end_sample=${MASTER_AUDIO_SAMPLES},asetpts=N/SR/TB" \
    -ac 1 -ar "$AUDIO_SAMPLE_RATE" -c:a pcm_s16le "$MASTER_AUDIO"

echo "============================================================"
echo "TBDub Student v2 - quality/stability"
echo "============================================================"
echo "Video:                  $VIDEO"
echo "Audio:                  $AUDIO"
echo "Output:                 $OUTPUT"
echo "Source frames:          $TOTAL_FRAMES"
echo "Source FPS:             $SOURCE_FPS"
echo "Source duration:        ${VIDEO_DURATION}s"
echo "Output chunks:          $NUM_CHUNKS / $FULL_NUM_CHUNKS"
echo "Output frames:          $OUTPUT_TOTAL_FRAMES"
echo "Output duration:        ${OUTPUT_DURATION}s"
echo "Core chunk frames:      $FRAMES_PER_CHUNK (${CHUNK_DURATION}s)"
echo "Context frames/side:    $CONTEXT_FRAMES"
echo "Max inference frames:   $((FRAMES_PER_CHUNK + 2 * CONTEXT_FRAMES))"
echo "Student steps:          $STUDENT_STEPS"
echo "Sigma shift:            $SIGMA_SHIFT"
echo "Motion from latents:    $MOTION_FROM_LATENTS"
echo "CPU offload:            enabled"
echo "Preserve upper face:    $PRESERVE_UPPER_FACE"
echo "Lower blend:            ${LOWER_FACE_BLEND_START} -> ${LOWER_FACE_BLEND_FULL}"
echo "Final audio duration:   ${MASTER_AUDIO_DURATION}s"
echo "Workdir:                $WORKDIR"
echo "============================================================"

CONCAT_FILE="$WORKDIR/concat.txt"
> "$CONCAT_FILE"

for ((i=0; i<NUM_CHUNKS; i++)); do
    CORE_START=$((i * FRAMES_PER_CHUNK))
    CORE_END=$((CORE_START + FRAMES_PER_CHUNK))
    (( CORE_END > OUTPUT_TOTAL_FRAMES )) && CORE_END=$OUTPUT_TOTAL_FRAMES
    CORE_COUNT=$((CORE_END - CORE_START))

    RUN_START=$((CORE_START - CONTEXT_FRAMES))
    (( RUN_START < 0 )) && RUN_START=0
    RUN_END=$((CORE_END + CONTEXT_FRAMES))
    (( RUN_END > TOTAL_FRAMES )) && RUN_END=$TOTAL_FRAMES
    RUN_COUNT=$((RUN_END - RUN_START))
    CORE_OFFSET=$((CORE_START - RUN_START))

    RUN_START_SAMPLE=$((RUN_START * SAMPLES_PER_FRAME))
    RUN_END_SAMPLE=$((RUN_END * SAMPLES_PER_FRAME))

    PART=$(printf "%03d" "$i")
    VIDEO_CHUNK="$WORKDIR/video_${PART}.mp4"
    AUDIO_CHUNK="$WORKDIR/audio_${PART}.wav"
    CHUNK_OUTPUT_DIR="$WORKDIR/inference_${PART}"
    REPORT_JSON="$WORKDIR/preprocess_${PART}.json"
    NORMALIZED_RUN="$WORKDIR/generated_run_${PART}.mp4"
    FINAL_CHUNK="$RESULTS_DIR/result_${PART}.mp4"
    mkdir -p "$CHUNK_OUTPUT_DIR"

    echo
    echo "============================================================"
    echo "Chunk $((i+1)) / $NUM_CHUNKS"
    echo "Core:    ${CORE_START}..$((CORE_END-1)) (${CORE_COUNT} frames)"
    echo "Run:     ${RUN_START}..$((RUN_END-1)) (${RUN_COUNT} frames)"
    echo "Offset:  $CORE_OFFSET frames of preroll context"
    echo "============================================================"

    echo "[1/5] Extracting frame-accurate video with context..."
    ffmpeg -y -hide_banner -loglevel error \
        -i "$VIDEO" -an \
        -vf "trim=start_frame=${RUN_START}:end_frame=${RUN_END},setpts=N/(${TARGET_FPS}*TB)" \
        -frames:v "$RUN_COUNT" -r "$TARGET_FPS" -fps_mode cfr \
        -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p "$VIDEO_CHUNK"

    VIDEO_CHUNK_FRAMES=$(get_frame_count "$VIDEO_CHUNK")
    (( VIDEO_CHUNK_FRAMES == RUN_COUNT )) || {
        echo "ERROR: Source chunk frame mismatch: expected=$RUN_COUNT got=$VIDEO_CHUNK_FRAMES"
        exit 1
    }

    echo "[2/5] Extracting exact audio samples with matching context..."
    ffmpeg -y -hide_banner -loglevel error \
        -i "$MASTER_AUDIO" -vn \
        -af "atrim=start_sample=${RUN_START_SAMPLE}:end_sample=${RUN_END_SAMPLE},asetpts=N/SR/TB" \
        -ac 1 -ar "$AUDIO_SAMPLE_RATE" -c:a pcm_s16le "$AUDIO_CHUNK"

    echo "[3/5] Running TBDub Student..."
    STUDENT_ARGS=(
        --video "$VIDEO_CHUNK"
        --audio "$AUDIO_CHUNK"
        --inference-mode student
        --num-student-steps "$STUDENT_STEPS"
        --sigma-shift "$SIGMA_SHIFT"
        --seed "$SEED"
        --preprocess-report "$REPORT_JSON"
        --output-dir "$CHUNK_OUTPUT_DIR"
        --cpu-offload
    )
    if [[ "$MOTION_FROM_LATENTS" == "1" ]]; then
        STUDENT_ARGS+=(--motion-from-latents)
    fi
    python inference.py "${STUDENT_ARGS[@]}"

    GENERATED=$(find "$CHUNK_OUTPUT_DIR" -type f -name "*.mp4" \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    [[ -n "$GENERATED" && -f "$GENERATED" ]] || {
        echo "ERROR: TBDub did not produce an MP4 for chunk $PART."
        exit 1
    }
    [[ -f "$REPORT_JSON" ]] || {
        echo "ERROR: TBDub did not write preprocessing report: $REPORT_JSON"
        exit 1
    }

    GENERATED_FRAMES=$(get_frame_count "$GENERATED")
    echo "      TBDub raw frames: $GENERATED_FRAMES; expected run frames: $RUN_COUNT"

    echo "[4/5] Normalizing Student output to exact run frame count..."
    ffmpeg -y -hide_banner -loglevel error \
        -i "$GENERATED" -an \
        -vf "setpts=N/(${TARGET_FPS}*TB),tpad=stop_mode=clone:stop=${RUN_COUNT},trim=end_frame=${RUN_COUNT},setpts=N/(${TARGET_FPS}*TB)" \
        -frames:v "$RUN_COUNT" -r "$TARGET_FPS" -fps_mode cfr \
        -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p "$NORMALIZED_RUN"

    NORMALIZED_FRAMES=$(get_frame_count "$NORMALIZED_RUN")
    (( NORMALIZED_FRAMES == RUN_COUNT )) || {
        echo "ERROR: Normalized run mismatch: expected=$RUN_COUNT got=$NORMALIZED_FRAMES"
        exit 1
    }

    echo "[5/5] Keeping core frames and stabilizing upper face..."
    if [[ "$PRESERVE_UPPER_FACE" == "1" ]]; then
        python "$BLEND_HELPER" \
            --source "$VIDEO_CHUNK" \
            --generated "$NORMALIZED_RUN" \
            --report "$REPORT_JSON" \
            --offset "$CORE_OFFSET" \
            --count "$CORE_COUNT" \
            --fps "$TARGET_FPS" \
            --blend-start "$LOWER_FACE_BLEND_START" \
            --blend-full "$LOWER_FACE_BLEND_FULL" \
            --x-fade "$LOWER_FACE_X_FADE" \
            --output "$FINAL_CHUNK"
    else
        ffmpeg -y -hide_banner -loglevel error \
            -i "$NORMALIZED_RUN" -an \
            -vf "trim=start_frame=${CORE_OFFSET}:end_frame=$((CORE_OFFSET + CORE_COUNT)),setpts=N/(${TARGET_FPS}*TB)" \
            -frames:v "$CORE_COUNT" -r "$TARGET_FPS" -fps_mode cfr \
            -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p "$FINAL_CHUNK"
    fi

    FINAL_CHUNK_FRAMES=$(get_frame_count "$FINAL_CHUNK")
    (( FINAL_CHUNK_FRAMES == CORE_COUNT )) || {
        echo "ERROR: Final core chunk mismatch: expected=$CORE_COUNT got=$FINAL_CHUNK_FRAMES"
        exit 1
    }

    ABS_CHUNK="$(realpath "$FINAL_CHUNK")"
    printf "file '%s'\n" "$ABS_CHUNK" >> "$CONCAT_FILE"

    rm -f "$VIDEO_CHUNK" "$AUDIO_CHUNK" "$NORMALIZED_RUN" "$REPORT_JSON"
    rm -rf "$CHUNK_OUTPUT_DIR"
    sync
    echo "Chunk $((i+1)) finished: $FINAL_CHUNK_FRAMES frames."
done

echo
echo "============================================================"
echo "Joining VIDEO-ONLY chunks..."
echo "============================================================"

JOINED_VIDEO="$WORKDIR/joined_video_only.mp4"
if ffmpeg -y -hide_banner -loglevel error \
    -f concat -safe 0 -i "$CONCAT_FILE" -an -c:v copy -movflags +faststart "$JOINED_VIDEO"; then
    echo "Joined without re-encoding."
else
    echo "Stream-copy concat failed; re-encoding join..."
    ffmpeg -y -hide_banner -loglevel error \
        -f concat -safe 0 -i "$CONCAT_FILE" -an \
        -vf "setpts=N/(${TARGET_FPS}*TB)" -r "$TARGET_FPS" -fps_mode cfr \
        -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
        -movflags +faststart "$JOINED_VIDEO"
fi

JOINED_FRAMES=$(get_frame_count "$JOINED_VIDEO")
if (( JOINED_FRAMES != OUTPUT_TOTAL_FRAMES )); then
    echo "WARNING: Joined=$JOINED_FRAMES, expected=$OUTPUT_TOTAL_FRAMES. Applying exact correction..."
    JOINED_FIXED="$WORKDIR/joined_video_fixed.mp4"
    ffmpeg -y -hide_banner -loglevel error \
        -i "$JOINED_VIDEO" -an \
        -vf "setpts=N/(${TARGET_FPS}*TB),tpad=stop_mode=clone:stop=${OUTPUT_TOTAL_FRAMES},trim=end_frame=${OUTPUT_TOTAL_FRAMES},setpts=N/(${TARGET_FPS}*TB)" \
        -frames:v "$OUTPUT_TOTAL_FRAMES" -r "$TARGET_FPS" -fps_mode cfr \
        -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
        -movflags +faststart "$JOINED_FIXED"
    JOINED_VIDEO="$JOINED_FIXED"
    JOINED_FRAMES=$(get_frame_count "$JOINED_VIDEO")
fi

(( JOINED_FRAMES == OUTPUT_TOTAL_FRAMES )) || {
    echo "ERROR: Animated video does not match requested output frame count. requested=$OUTPUT_TOTAL_FRAMES animated=$JOINED_FRAMES"
    exit 1
}

echo "Frame-count check passed: requested=$OUTPUT_TOTAL_FRAMES animated=$JOINED_FRAMES"

echo
echo "============================================================"
echo "Muxing ONE continuous audio track..."
echo "============================================================"

ffmpeg -y -hide_banner -loglevel error \
    -i "$JOINED_VIDEO" -i "$MASTER_AUDIO" \
    -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k \
    -movflags +faststart "$OUTPUT"

FINAL_FRAMES=$(get_frame_count "$OUTPUT")
FINAL_FPS=$(get_fps "$OUTPUT")
(( FINAL_FRAMES == OUTPUT_TOTAL_FRAMES )) || {
    echo "ERROR: Frame count changed during final mux. requested=$OUTPUT_TOTAL_FRAMES final=$FINAL_FRAMES"
    exit 1
}

# FeatherTalk-style validation.
VALIDATION_WAV="$WORKDIR/feathertalk_audio_check.wav"
ffmpeg -y -hide_banner -loglevel error \
    -i "$OUTPUT" -vn -f wav -ar "$AUDIO_SAMPLE_RATE" "$VALIDATION_WAV"

DECODED_SAMPLES=$(ffprobe -v error -select_streams a:0 \
    -show_entries stream=duration_ts \
    -of default=noprint_wrappers=1:nokey=1 "$VALIDATION_WAV" | head -n 1)

if [[ -z "$DECODED_SAMPLES" || "$DECODED_SAMPLES" == "N/A" || ! "$DECODED_SAMPLES" =~ ^[0-9]+$ ]]; then
    DECODED_DURATION=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$VALIDATION_WAV")
    DECODED_SAMPLES=$(awk -v d="$DECODED_DURATION" -v sr="$AUDIO_SAMPLE_RATE" 'BEGIN { printf "%d", d*sr + 0.5 }')
fi

DECODED_DURATION=$(awk -v s="$DECODED_SAMPLES" -v sr="$AUDIO_SAMPLE_RATE" 'BEGIN { printf "%.6f", s/sr }')
if (( DECODED_SAMPLES >= 400 )); then
    HUBERT_TOKENS=$(( (DECODED_SAMPLES - 80) / 320 ))
else
    HUBERT_TOKENS=0
fi
EVEN_HUBERT_TOKENS=$((HUBERT_TOKENS - (HUBERT_TOKENS % 2)))
FEATHERTALK_AUDIO_FRAMES=$((EVEN_HUBERT_TOKENS / 2))
EXTRA_AUDIO_FRAMES=$((FEATHERTALK_AUDIO_FRAMES - OUTPUT_TOTAL_FRAMES))

(( FEATHERTALK_AUDIO_FRAMES >= OUTPUT_TOTAL_FRAMES )) || {
    echo "ERROR: Final audio is too short for FeatherTalk. video=$OUTPUT_TOTAL_FRAMES audio_frames=$FEATHERTALK_AUDIO_FRAMES"
    exit 1
}
(( EXTRA_AUDIO_FRAMES <= 3 )) || {
    echo "ERROR: Final audio is unexpectedly longer than video. extra=$EXTRA_AUDIO_FRAMES"
    exit 1
}

echo
echo "============================================================"
echo "VALIDATION PASSED"
echo "============================================================"
echo "Silent/source video frames:    $TOTAL_FRAMES"
echo "Requested output frames:       $OUTPUT_TOTAL_FRAMES"
echo "Animated/final video frames:   $FINAL_FRAMES"
echo "Final video FPS:               $FINAL_FPS"
echo "FeatherTalk-decoded WAV:       ${DECODED_DURATION}s"
echo "Estimated audio feature frames:$FEATHERTALK_AUDIO_FRAMES"
echo "Extra audio feature frames:    $EXTRA_AUDIO_FRAMES"
echo "Student config:                steps=$STUDENT_STEPS sigma_shift=$SIGMA_SHIFT motion_from_latents=$MOTION_FROM_LATENTS"
echo "Context frames/side:           $CONTEXT_FRAMES"
echo "NUM_ITERATIONS:                $NUM_CHUNKS"
echo "Preserve upper face:           $PRESERVE_UPPER_FACE"
echo "Output:                        $OUTPUT"
echo "============================================================"
