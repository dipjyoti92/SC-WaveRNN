from utils.dataset import get_vocoder_datasets
from utils.dsp import *
from models.fatchord_version_spk_embed import WaveRNN
from utils.paths import Paths
from utils.display import simple_table
import torch
import argparse
from encoder.params_model import model_embedding_size as speaker_embedding_size
from encoder import inference as encoder
from pathlib import Path

def gen_testset(model, test_set, samples, batched, target, overlap, save_path) :

    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):

        if i > samples : break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL' :
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else :
            x = label_2_float(x, bits)

        save_wav(x, f'{save_path}{k}k_steps_{i}_target.wav')

        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = f'{save_path}{k}k_steps_{i}_{batch_str}.wav'

        _ = model.generate(m, save_str, batched, target, overlap, hp.mu_law)


def gen_from_file(model, load_path, enc_model_fpath, save_path, batched, target, overlap) :

    k = model.get_step() // 1000
    file_name = load_path.split('/')[-1]

    wav = load_wav(load_path)
    #save_wav(wav, f'{save_path}__{file_name}__{k}k_steps_target.wav')

    encoder.load_model(enc_model_fpath)
    preprocessed_wav = encoder.preprocess_wav(load_path)
    embed = encoder.embed_utterance(preprocessed_wav)
    spk_embd = torch.tensor(embed).unsqueeze(0)

    mel = melspectrogram(wav)
    mel = torch.tensor(mel).unsqueeze(0)

    batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
    #save_str = f'{save_path}__{file_name}__{k}k_steps_{batch_str}_spk_embed.wav'
    save_str = f'{file_name}'
    _ = model.generate(mel, spk_embd, save_str, batched, target, overlap, hp.mu_law)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate WaveRNN Samples')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slow Unbatched Generation')
    parser.add_argument('--samples', '-s', type=int, help='[int] number of utterances to generate')
    parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    parser.add_argument('--file', '-f', type=str, help='[string/path] for testing a wav outside dataset')
    parser.add_argument('--weights', '-w', type=str, help='[string/path] checkpoint file to load weights from')
    parser.add_argument('--gta', '-g', dest='use_gta', action='store_true', help='Generate from GTA testset')
    parser.add_argument("-e", "--enc_model_fpath", type=Path, default="encoder/saved_models/pretrained.pt",help="Path to a saved encoder")
    parser.add_argument("--output", "-out", type=Path, help="output path")


    parser.set_defaults(batched=hp.voc_gen_batched)
    parser.set_defaults(samples=hp.voc_gen_at_checkpoint)
    parser.set_defaults(target=hp.voc_target)
    parser.set_defaults(overlap=hp.voc_overlap)
    parser.set_defaults(file=None)
    parser.set_defaults(weights=None)
    parser.set_defaults(gta=False)

    args = parser.parse_args()

    batched = args.batched
    samples = args.samples
    target = args.target
    overlap = args.overlap
    enc_path = args.enc_model_fpath
    file = args.file
    gta = args.gta
    out = args.output

    print('\nInitialising Model...\n')

    model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.num_mels,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    mode=hp.voc_mode).cuda()

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    restore_path = args.weights if args.weights else paths.voc_latest_weights

    model.restore(restore_path)

    simple_table([('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', target if batched else 'N/A'),
                  ('Overlap Samples', overlap if batched else 'N/A')])

    if file :
        gen_from_file(model, file, enc_path, out, batched, target, overlap)
    else :
        _, test_set = get_vocoder_datasets(paths.data, 1, gta)
        gen_testset(model, test_set, samples, batched, target, overlap, paths.voc_output)

    print('\n\nExiting...\n')
