# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from tacotron2.text import text_to_sequence
import models
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

import sys
import csv

import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from apex import amp

from utils import write_hdf5
from decode import make_wav

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, default='text.txt',
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', default="parallel_output",
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--suffix', type=str, default="", help="output filename suffix")
    parser.add_argument('--tacotron2', type=str,default='gan_checkpoint/checkpoint_Tacotron2_1500',
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--waveglow', type=str,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--amp-run', default=True, action='store_true',
                        help='inference with AMP')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--include-warmup', default=True,action='store_true',
                        help='Include warmup')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    #---------------------------------------------------------------------------------------
    #여기부터 parallel_wavegan관련
    parser.add_argument("--feats-scp", "--scp", default=None, type=str,
                        help="kaldi-style feats.scp file. "
                             "you need to specify either feats-scp or dumpdir.")

    parser.add_argument("--dumpdir", default='mel', type=str,
                        help="directory including feature files. "
                             "you need to specify either feats-scp or dumpdir. .h5파일 넣어주는 부분")
    #'../../egs/kss/voc1/dump/dev/test.1'
    parser.add_argument("--outdir", default='parallel_output', type=str,
                        help="directory to save generated speech.")
    parser.add_argument("--checkpoint", default='parallel_wavegan_checkpoint/checkpoint-110125steps.pkl', type=str,
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", default=None, type=str,
                        help="yaml format configuration file. if not explicitly provided, "
                             "it will be searched in the checkpoint directory. (default=None)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    #-----------------------------------------------------------------------------------------
    return parser


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def load_and_setup_model(model_name, parser, checkpoint, amp_run, forward_is_infer=False):
    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, to_cuda=True,
                             forward_is_infer=forward_is_infer)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)

        model.load_state_dict(state_dict)

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)

    model.eval()

    if amp_run:
        model.half()

    return model


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts):

    d = []
    for i,text in enumerate(texts):
        d.append(torch.IntTensor(
            text_to_sequence(text, ['english_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if torch.cuda.is_available():
        text_padded = torch.autograd.Variable(text_padded).cuda().long()
        input_lengths = torch.autograd.Variable(input_lengths).cuda().long()
    else:
        text_padded = torch.autograd.Variable(text_padded).long()
        input_lengths = torch.autograd.Variable(input_lengths).long()

    return text_padded, input_lengths

def make_script():
    g_csv = 'nlp_script_dataset2.csv'

    f = open(g_csv, 'r', encoding='cp949')
    lines = csv.reader(f)
    lines = list(lines)[37]

    script = lines[1].split('  ')[1:]
    print(script)
    f.close()

    return script[:5]

class MeasureTime():
    def __init__(self, measurements, key):
        self.measurements = measurements
        self.key = key

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter() - self.t0


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT,
                                              args.output+'/'+args.log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     args.amp_run, forward_is_infer=True)

    jitted_tacotron2 = torch.jit.script(tacotron2)



    #texts = []
    #texts.append(make_script())


    try:
        print(args.input)
        f = open(args.input, 'r')
        texts = f.readlines()
    except:
        print("Could not read file")
        sys.exit(1)

    #데이터 불러 와 주고 요기 아래부터 for 문으로 돌려주면 될 듯.
    
    if args.include_warmup:
        sequence = torch.randint(low=0, high=80, size=(1,50),
                                 dtype=torch.long).cuda()
        input_lengths = torch.IntTensor([sequence.size(1)]).cuda().long()
        for i in range(3):
            with torch.no_grad():
                mel, mel_lengths, _ = jitted_tacotron2(sequence, input_lengths)


    measurements = {}

    sequences_padded, input_lengths = prepare_input_sequence(texts)

    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time"):
        mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths)
        #-------------------------------------------------------------- 
        #.h5모델로 멜 만드는 부분
        #print(mel.float().data.cpu().numpy())
        write_hdf5('mel/sw_embedded.h5', "feats", mel.float().data.cpu().numpy())
        #--------------------------------------------------------------
        
    print("Stopping after", mel.size(2), "decoder steps")
    #여기있는게 원래 코드


    '''
    #-------------------------------------------------------------------------------------------------------
    texts_list = make_script()
    for idx, texts in enumerate(texts_list):
        text = []
        text.append(''.join(texts))
        print(text)
        if args.include_warmup:
            sequence = torch.randint(low=0, high=80, size=(1, 50),
                                     dtype=torch.long).cuda()
            input_lengths = torch.IntTensor([sequence.size(1)]).cuda().long()
            for i in range(3):
                with torch.no_grad():
                    mel, mel_lengths, _ = jitted_tacotron2(sequence, input_lengths)

        measurements = {}

        sequences_padded, input_lengths = prepare_input_sequence(text)

        with torch.no_grad(), MeasureTime(measurements, "tacotron2_time"):
            mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths)
            # --------------------------------------------------------------
            # .h5모델로 멜 만드는 부분
            # print(mel.float().data.cpu().numpy())
            write_hdf5('mel/test2_{}.h5'.format(idx), "feats", mel.float().data.cpu().numpy())
            # --------------------------------------------------------------

        print("Stopping after", mel.size(2), "decoder steps")
    #-------------------------------------------------------------------------------------------------------
    '''


    DLLogger.flush()
    make_wav(args)
    

if __name__ == '__main__':
    main()