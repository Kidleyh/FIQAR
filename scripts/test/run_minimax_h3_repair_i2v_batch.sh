#!/usr/bin/env bash
set -uo pipefail

ROOT="/gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio-minimaxh3"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/data/repair}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/repair}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/py312/bin/python}"
I2V_SCRIPT="${I2V_SCRIPT:-${ROOT}/scripts/test/minimax_h3_single_image_i2v.py}"

WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
NUM_FRAMES="${NUM_FRAMES:-175}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
SEED="${SEED:-0}"
VRAM_LIMIT_GB="${VRAM_LIMIT_GB:-60}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
INCLUDE_EXTRA_MOTIONS="${INCLUDE_EXTRA_MOTIONS:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_JOBS="${MAX_JOBS:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
GENERATION_WIDTH=$(( (WIDTH + 31) / 32 * 32 ))
GENERATION_HEIGHT=$(( (HEIGHT + 31) / 32 * 32 ))

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -d "${INPUT_ROOT}" ]]; then
    echo "[fatal] input root does not exist: ${INPUT_ROOT}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[fatal] Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${I2V_SCRIPT}" ]]; then
    echo "[fatal] I2V script does not exist: ${I2V_SCRIPT}" >&2
    exit 1
fi
if [[ ! "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[fatal] NUM_SHARDS must be a positive integer; got ${NUM_SHARDS}" >&2
    exit 1
fi
if [[ ! "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || [[ "${SHARD_INDEX}" -ge "${NUM_SHARDS}" ]]; then
    echo "[fatal] SHARD_INDEX must be in [0, NUM_SHARDS); got ${SHARD_INDEX}/${NUM_SHARDS}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"
STATUS_FILE="${OUTPUT_ROOT}/batch_status_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.tsv"
printf 'timestamp\tstatus\tscene\timage\tmotion\toutput\n' > "${STATUS_FILE}"
echo "[shard] index=${SHARD_INDEX} total=${NUM_SHARDS}"

motion_ids=(
    "left_to_right"
    "right_to_left"
    "left_to_right_closer"
    "right_to_left_closer"
)
motion_texts=(
    "The camera performs a pronounced, continuous left-to-right lateral dolly movement over a substantial physical distance. Begin from a clearly left-side viewpoint and finish at a distinctly different right-side viewpoint. Maintain a steady speed and produce strong, physically correct parallax. This must be a wide tracking movement, not a subtle drift."
    "The camera performs a pronounced, continuous right-to-left lateral dolly movement over a substantial physical distance. Begin from a clearly right-side viewpoint and finish at a distinctly different left-side viewpoint. Maintain a steady speed and produce strong, physically correct parallax. This must be a wide tracking movement, not a subtle drift."
    "The camera performs a pronounced left-to-right lateral dolly while simultaneously moving forward toward the landmark. Travel across a wide lateral baseline and close the physical distance substantially, so the landmark becomes noticeably larger through real camera translation rather than zooming. Produce strong, physically correct parallax."
    "The camera performs a pronounced right-to-left lateral dolly while simultaneously moving forward toward the landmark. Travel across a wide lateral baseline and close the physical distance substantially, so the landmark becomes noticeably larger through real camera translation rather than zooming. Produce strong, physically correct parallax."
)

if [[ "${INCLUDE_EXTRA_MOTIONS}" == "1" ]]; then
    motion_ids+=("forward_dolly_closer" "left_to_right_farther")
    motion_texts+=(
        "The camera performs a strong, continuous forward dolly toward the landmark over a substantial physical distance. The landmark grows noticeably larger because the camera truly approaches it, with increasingly strong foreground parallax. This is a decisive forward camera move, not a zoom and not a subtle drift."
        "The camera performs a pronounced left-to-right lateral dolly while simultaneously pulling backward away from the landmark. Travel across a wide lateral baseline and increase the physical distance substantially, so the landmark becomes noticeably smaller through real camera translation rather than zooming. Produce strong, physically correct parallax."
    )
fi

scene_subject() {
    case "$1" in
        brandenburg_gate)
            printf '%s' "The Brandenburg Gate in Berlin, a monumental neoclassical sandstone landmark with six large columns, accurate architectural proportions, sharp structural details, and realistic fine stone textures"
            ;;
        sacre_coeur)
            printf '%s' "The Basilica of Sacre-Coeur in Paris, a monumental white-stone basilica with accurate domes, arches, steps, architectural proportions, and finely resolved masonry details"
            ;;
        taj_mahal)
            printf '%s' "The Taj Mahal in Agra, a monumental white-marble mausoleum with an accurate central dome, minarets, arches, symmetry, architectural proportions, and finely resolved marble details"
            ;;
        temple_nara_japan)
            printf '%s' "A historic Buddhist temple in Nara, Japan, with accurate traditional wooden architecture, roof geometry, structural proportions, fine timber details, and surrounding grounds"
            ;;
        *)
            printf '%s' "The landmark shown in the input image, preserving its exact identity, architecture, geometry, materials, and surrounding scene"
            ;;
    esac
}

total=0
completed=0
skipped=0
failed=0
stop_requested=0
global_task_index=0

while IFS= read -r -d '' scene_dir; do
    scene_name="$(basename "${scene_dir}")"
    subject="$(scene_subject "${scene_name}")"

    while IFS= read -r -d '' input_image; do
        image_name="$(basename "${input_image}")"
        image_stem="${image_name%.*}"
        case_output_dir="${OUTPUT_ROOT}/${scene_name}/${image_stem}"
        case_log_dir="${OUTPUT_ROOT}/logs/${scene_name}/${image_stem}"
        mkdir -p "${case_output_dir}" "${case_log_dir}"

        for motion_index in "${!motion_ids[@]}"; do
            motion_id="${motion_ids[${motion_index}]}"
            motion_text="${motion_texts[${motion_index}]}"
            output_video="${case_output_dir}/${motion_id}_${NUM_FRAMES}f_${GENERATION_WIDTH}x${GENERATION_HEIGHT}.mp4"
            output_json="${output_video%.mp4}.json"
            log_path="${case_log_dir}/${motion_id}_${NUM_FRAMES}f_${GENERATION_WIDTH}x${GENERATION_HEIGHT}.log"

            current_task_index=${global_task_index}
            global_task_index=$((global_task_index + 1))
            if [[ $((current_task_index % NUM_SHARDS)) -ne "${SHARD_INDEX}" ]]; then
                continue
            fi

            if [[ "${MAX_JOBS}" -gt 0 && "${total}" -ge "${MAX_JOBS}" ]]; then
                stop_requested=1
                break
            fi
            total=$((total + 1))

            if [[ "${DRY_RUN}" != "1" && -s "${output_video}" && -s "${output_json}" ]]; then
                echo "[skip-existing] scene=${scene_name} image=${image_name} motion=${motion_id}"
                printf '%s\tSKIPPED\t%s\t%s\t%s\t%s\n' \
                    "$(date '+%F %T')" "${scene_name}" "${image_name}" "${motion_id}" "${output_video}" >> "${STATUS_FILE}"
                skipped=$((skipped + 1))
                continue
            fi

            prompt="<SUBJECT>: ${subject}. <SCENE>: A photorealistic daytime view of the landmark and its surrounding plaza or grounds. Preserve the landmark's identity and keep its architecture rigid, sharp, geometrically accurate, and temporally consistent throughout the entire video. The foreground ground surface must remain highly detailed and stable, with realistic fine pavement or terrain textures, sharp seams and edges, coherent surface patterns, consistent perspective, natural contact shadows, and no texture flickering or swimming. Maintain stable daylight, exposure, colors, materials, and fine details across all frames. <EVENT>: ${motion_text} Keep the landmark recognizable, sharp, and structurally stable for the full shot. No artificial zoom, no static camera, no tiny motion, no camera shake, no sudden acceleration, no jumps, no blur, no flickering, no texture swimming, no crawling ground patterns, no warped ground, no structural deformation, no duplicated architectural elements, and no objects appearing or disappearing. The video is silent."

            cmd=(
                "${PYTHON_BIN}" -u "${I2V_SCRIPT}"
                --input-image "${input_image}"
                --output-video "${output_video}"
                --width "${WIDTH}"
                --height "${HEIGHT}"
                --num-frames "${NUM_FRAMES}"
                --num-inference-steps "${NUM_INFERENCE_STEPS}"
                --seed "${SEED}"
                --cfg-scale 1.0
                --flow-shift 12.0
                --audio-flow-shift 3.0
                --vram-limit-gb "${VRAM_LIMIT_GB}"
                --no-audio
                --keep-aligned-output
                --prompt "${prompt}"
            )
            if [[ "${DRY_RUN}" == "1" ]]; then
                cmd+=(--dry-run)
            else
                cmd+=(--overwrite)
            fi

            echo "[run] shard=${SHARD_INDEX}/${NUM_SHARDS} local=${total} global_index=${current_task_index} scene=${scene_name} image=${image_name} motion=${motion_id}"
            "${cmd[@]}" 2>&1 | tee "${log_path}"
            rc=${PIPESTATUS[0]}
            if [[ "${rc}" -eq 0 ]]; then
                status="DRY_RUN_OK"
                if [[ "${DRY_RUN}" != "1" ]]; then
                    status="COMPLETED"
                fi
                completed=$((completed + 1))
            else
                status="FAILED_${rc}"
                failed=$((failed + 1))
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date '+%F %T')" "${status}" "${scene_name}" "${image_name}" "${motion_id}" "${output_video}" >> "${STATUS_FILE}"
        done

        if [[ "${stop_requested}" -eq 1 ]]; then
            break
        fi
    done < <(
        find "${scene_dir}" -maxdepth 1 -type f \
            \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) \
            -print0 | sort -z
    )

    if [[ "${stop_requested}" -eq 1 ]]; then
        break
    fi
done < <(find "${INPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo "[done] shard=${SHARD_INDEX}/${NUM_SHARDS} assigned=${total} completed=${completed} skipped=${skipped} failed=${failed} status=${STATUS_FILE}"
if [[ "${failed}" -gt 0 ]]; then
    exit 1
fi
