import os
import cv2
import numpy as np
import argparse
import sys
from multiprocessing import Pool

def _draw_block_process(args):
    frame, pixel_size, font, font_scale, thick, row_offsets, start_row, end_row, text_every = args
    h, w, _ = frame.shape
    block_img = np.zeros(((end_row - start_row) * pixel_size, w * pixel_size, 3), dtype=np.uint8)
    for idx_i, i in enumerate(range(start_row, end_row)):
        y0 = idx_i * pixel_size
        for j in range(w):
            x0 = j * pixel_size
            b, g, r = frame[i, j]
            color = (int(b), int(g), int(r))
            block_img[y0:y0 + pixel_size, x0:x0 + pixel_size] = color

            # Draw text only every `text_every` pixels
            if (i % text_every == 0) and (j % text_every == 0):
                bright = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = (255, 255, 255) if bright < 128 else (0, 0, 0)
                for k, val in enumerate((r, g, b)):
                    y_text = y0 + row_offsets[k]
                    cv2.putText(block_img, str(val), (x0 + 1, y_text),
                                font, font_scale, text_color, thick, cv2.LINE_AA)
    return (start_row, block_img)

def process_frame_multiprocess(frame, pixel_size, font, font_scale, thick, row_offsets, num_blocks=4, text_every=16):
    h, w, _ = frame.shape
    rows_per_block = (h + num_blocks - 1) // num_blocks
    blocks = [(i * rows_per_block, min((i + 1) * rows_per_block, h)) for i in range(num_blocks)]

    args_list = [(frame, pixel_size, font, font_scale, thick, row_offsets, start, end, text_every)
                 for start, end in blocks]

    with Pool(processes=num_blocks) as pool:
        results = pool.map(_draw_block_process, args_list)

    results.sort(key=lambda x: x[0])
    out_img = np.vstack([block for _, block in results])
    return out_img

def extract_rgb_frames(video_path, pixel_size=16, start=0, end=None, num_blocks=4, text_every=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None or end > total_frames - 1:
        end = total_frames - 1

    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = f"{name}_color_frames"
    os.makedirs(out_dir, exist_ok=True)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = pixel_size / 75
    thick = 1
    row_offsets = [(k + 1) * (pixel_size // 4) for k in range(3)]

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count < start:
            count += 1
            continue
        if count > end:
            break

        out_img = process_frame_multiprocess(
            frame, pixel_size, font, font_scale, thick, row_offsets, num_blocks, text_every
        )
        cv2.imwrite(
            os.path.join(out_dir, f"frame_{count:04d}.png"), out_img
        )

        progress = (count - start + 1) / (end - start + 1) * 100
        sys.stdout.write(
            f"\rProcessing frame {count}/{end} ({progress:.1f}%)"
        )
        sys.stdout.flush()
        count += 1

    cap.release()
    print(f"\nFinished. Frames saved in '{out_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract RGB frames as pixel art with text overlay (multiprocessing)"
    )
    parser.add_argument("filename", help="Path to input video file")
    parser.add_argument("-p", "--pixel", type=int, default=16, help="Pixel size for blocks")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start frame index")
    parser.add_argument("-e", "--end", type=int, help="End frame index")
    parser.add_argument("-b", "--blocks", type=int, default=4, help="Number of process blocks (default 4)")
    parser.add_argument("-t", "--text_every", type=int, default=16, help="Draw text every N pixels")
    args = parser.parse_args()

    extract_rgb_frames(
        args.filename,
        pixel_size=args.pixel,
        start=args.start,
        end=args.end,
        num_blocks=args.blocks,
        text_every=args.text_every
    )
