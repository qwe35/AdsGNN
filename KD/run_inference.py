import os
import argparse
import torch
from utils import *

def run_model_inference(infile, outfile, pid, model_dir, batch_size, log_steps, max_seq_len):
    gpu_count = torch.cuda.device_count()
    DEVICES_ARGS = ""
    if gpu_count > 0 and pid < gpu_count:
        DEVICES_ARGS = str(pid)
    env_cmd = "" if os.name == "nt" else f"CUDA_VISIBLE_DEVICES='{DEVICES_ARGS}'"
    cmd = f"{env_cmd} python main.py " \
          f"--mode inference --model_dir {model_dir} --max_seq_len {max_seq_len} " \
          f"--inference_input_path {infile} --inference_output_path {outfile} " \
          f"--transformer_pretrained_weights dummy --batch_size {batch_size} --log_steps {log_steps} "
    print(cmd)
    assert os.system(cmd) == 0, f'Command failed!: {cmd}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=None, type=str, required=True,
                        help="The input file.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The model directory.")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="The output file.")
    parser.add_argument("--n_process", default=0, type=int,
                         help="Number of processes to run.")
    parser.add_argument("--batch_size", default=2048, type=int,
                         help="Prediction batch size.")
    parser.add_argument("--max_line_per_file", default=-1, type=int,
                         help="max line count for each split of file.")
    parser.add_argument("--log_steps", default=10000, type=int,
                         help="log steps.")
    parser.add_argument("--input_file_ext", default="txt", type=str,
                        help="If input is a directory, specify the file extension to search for in the directory.")
    parser.add_argument("--output_file_ext", default="pred", type=str,
                        help="If output is a directory, specify the file extension to use for the output files.")
    parser.add_argument("--max_seq_len", default=64, type=int, help="max seq len")
    args = parse_aether_args(parser)
    processor_count = get_processor_count(args.n_process)
    file_pairs = get_file_pairs(args.input_path, args.output_path, args.input_file_ext, args.output_file_ext)

    kwargs = {
        "model_dir": args.model_dir,
        "batch_size": args.batch_size,
        "log_steps": args.log_steps,
        "max_seq_len": args.max_seq_len
    }
    for infile, outfile in file_pairs:
        print(f"\nRunning for input file: {infile}, output to: {outfile}")
        if processor_count <= 1:
            run_model_inference(infile, outfile, 0, **kwargs)
        else:
            map_reduce(infile, outfile, run_model_inference, n_process=processor_count, max_line_per_file=args.max_line_per_file, **kwargs)

if __name__ == '__main__':
    main()