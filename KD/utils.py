import os
import re
import math
import collections
import glob
import traceback
from tqdm import tqdm
import multiprocessing as mp

def get_line_count(infile):
    counter = 0
    for _ in tqdm(open(infile, 'r', encoding='utf-8'), desc=f'counting lines for {infile}'):
        counter += 1
    return counter

def isalphanum(ch):
    if ch >= '0' and ch <= '9':
        return True
    if ch >= 'a' and ch <= 'z':
        return True
    if ch >= 'A' and ch <= 'Z':
        return True
    return False

def find_index(str, substr, start=0, end=None):
    if end is None:
        end = len(str)
    if start < 0 or start >= len(str) or end < 0 or end > len(str) or start >= end:
        return -1
    try:
        return str.index(substr, start, end)
    except:
        return -1

# normalizing text by only adding spaces before and after punctuations
def split_on_punct(text):
    text = text.replace('\xa0', ' ')
    new_chars = []
    for i, ch in enumerate(text):
        if ch.isalnum() or ch.isspace():
            new_chars.append(ch)
        else:
            if i > 0 and (text[i-1].isalnum() or text[i-1] != ch):
                new_chars.append(' ')
            new_chars.append(ch)
            if i < len(text) - 1 and text[i+1].isalnum():
                new_chars.append(' ')
    new_text = ''.join(new_chars)
    return re.sub(r'\s+', ' ', new_text).strip()

def get_processor_count(input_processor_count):
    max_processor_count = mp.cpu_count()
    if input_processor_count <= 0:
        processor_count = max_processor_count
    else:
        processor_count = min(input_processor_count, max_processor_count)
    print(f"\nSetting n_process to {processor_count}\n")
    return processor_count

def get_file_pairs(input_path, output_path, input_file_ext=None, output_file_ext=None):
    assert os.path.exists(input_path), f"Input path does not exist: {input_path}"
    file_pairs = []
    if os.path.isdir(input_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for infile in glob.glob(os.path.join(input_path, f"*.{input_file_ext}")):
            basename = os.path.basename(infile)
            first, ext = os.path.splitext(basename)
            outfile = os.path.join(output_path, f"{first}.{output_file_ext}")
            file_pairs.append((infile, outfile))
    else:
        file_pairs.append((input_path, output_path))
    assert len(file_pairs) > 0, f"No valid input found for path: {input_path}!"
    return file_pairs

################################
# AEther tools
################################
def parse_aether_args(parser):
    parser.add_argument("--ExperimentID", default=None, type=str)
    parser.add_argument("--Owner", default=None, type=str)
    parser.add_argument("--Priority", default=3, type=int)
    parser.add_argument("--NodeID", default=None, type=str)

    args, unknown = parser.parse_known_args()
    for u in unknown:
        k, v = u.split("=")
        setattr(args, k, v)
    return args

################################
# File IO
################################
class Writer:
    def __init__(self, file, encoding='utf-8'):
        self.file = file
        self.encoding = encoding
        self.writer = None
    
    def __enter__(self):
        self.writer = open(self.file, 'w', encoding=self.encoding)
        return self.writer

    def __exit__(self, type, value, traceback):
        if self.writer is not None:
            self.writer.close()

class TsvReader():
    def __init__(self, filename, separator='\t', column_check='ignore', encoding='utf-8', names=None, strip=True, column_count=0, raw_line_field='raw_line', infile_error_handling=None):
        self.filename = filename
        self.column_count = column_count
        self.sep = separator
        self.encoding = encoding
        self.column_check = column_check
        self.names = names
        self.strip = strip
        self.raw_line_field = raw_line_field
        self.infile_error_handling = infile_error_handling

        if self.names is not None:
            assert isinstance(self.names, list)
            self.column_count = len(self.names)

    @staticmethod
    def get_type_name(filename):
        basename = os.path.basename(filename)
        return basename.split('.', 1)[0]

    def __iter__(self):
        with open(self.filename, 'r', encoding=self.encoding, errors=self.infile_error_handling) as reader:
            custom_type = None
            if isinstance(self.names, list) and self.column_count > 0:
                typename = self.get_type_name(self.filename)
                field_names = ' '.join(self.names)
                if self.raw_line_field is not None:
                    field_names += ' ' + self.raw_line_field
                custom_type = collections.namedtuple(typename, field_names)
            for line in reader:
                items = line.split(self.sep)
                if self.strip:
                    items = [x.strip() for x in items]
                if self.column_count > 0 and len(items) != self.column_count:
                    info = "Columns count not equal to {}: {}".format(self.column_count, line)
                    if self.column_check == 'ignore':
                        continue
                    elif self.column_check == 'debug':
                        print(info)
                    elif self.column_check == 'raise':
                        raise Exception(info)
                if custom_type is None:
                    yield items
                else:
                    if self.raw_line_field is not None:
                        items.append(line)
                    yield custom_type(*items)

################################
# Multi-processing tools
################################
# split input file into multiple files
def mr_split_files(infile, n_split, max_line_per_file):
    line_count = get_line_count(infile)
    max_per_file = math.ceil(line_count / n_split)
    if max_line_per_file > 0:
        max_per_file = min(max_per_file, max_line_per_file)
    #print(f'max line per file: {max_per_file}')
    idx = 0
    writer = None
    file_names = []
    for i, line in tqdm(enumerate(open(infile, 'r', encoding='utf-8')), desc='split file'):
        if i % max_per_file == 0:
            if writer is not None:
                writer.close()
            file_name = f'{infile}.{idx}'
            file_names.append(file_name)
            print(f'writing {file_name}...')
            writer = open(file_name, 'w', encoding='utf-8')
            idx += 1
        writer.write(line)
    if writer is not None:
        writer.close()
    return file_names

class CustomProcess(mp.Process):
    """
    Class which returns child Exceptions to Parent.
    https://stackoverflow.com/a/33599967/4992248
    """
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._parent_conn, self._child_conn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._child_conn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            # raise e  # You can still raise this exception if you need to

    @property
    def exception(self):
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception

def mr_run(infiles, outfile, base_idx, func, *args, **kwargs):
    processes = []
    outfile_names = []
    for i, infile_split in enumerate(infiles):
        outfile_split = f'{outfile}.{base_idx+i}'
        outfile_names.append(outfile_split)
        p = CustomProcess(target=func, args=(infile_split, outfile_split, i, *args), kwargs=kwargs)
        print('starting process {} with input {} and output {}'.format(i, infile_split, outfile_split))
        p.start()
        processes.append(p)
    # wait for all processes
    for p in processes:
        p.join()
        if p.exception:
            error, traceback = p.exception
            raise Exception("Child exception: {}, {}".format(error, traceback))
    return outfile_names

# merge multiple files into one
def mr_merge_files(infiles, outfile):
    with Writer(outfile) as writer:
        for outfile_split in tqdm(infiles):
            for line in open(outfile_split, 'r', encoding='utf-8'):
                if not line.strip():
                    continue
                writer.write(line)

def mr_clean_up(*file_lists):
    for file_list in file_lists:
        if isinstance(file_list, list):
            for file in file_list:
                os.remove(file)
        else:
            os.remove(file)

# running multiple process by splitting the input file into multiple files
# the first three arguments of the function `func` must be: infile, outfile, process_id
def map_reduce(infile, outfile, func, n_process, *args, **kwargs):
    max_line_per_file = kwargs.pop('max_line_per_file') if 'max_line_per_file' in kwargs else 5000000
    # 1. split input file
    infile_names = mr_split_files(infile, n_process, max_line_per_file)
    outfile_names = []
    try:
        for i in range(0, len(infile_names), n_process):
            start = i
            end = min(i+n_process, len(infile_names))
            print(f"running batch {start} to {end}")
            # 2. run func with each split of the file
            part_outfile_names = mr_run(infile_names[start:end], outfile, start, func, *args, **kwargs)
            outfile_names.extend(part_outfile_names)
        # 3. merge results
        print('merging results...')
        mr_merge_files(outfile_names, outfile)
    finally:
        # 4. clean up
        print('clean up...')
        mr_clean_up(infile_names, outfile_names)
