import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='either imagenet, kodak, cub200, stanford_dogs')
parser.add_argument('--compression', required=True, help='one of bpg, jpeg, webp, rnn, cpm')
parser.add_argument('--records', required=False, help='records file with encoded image data')
parser.add_argument('--rnn_ckpt_dir', required=False, help='dir containing subfolder with checkpoints and config.json')
parser.add_argument('--bpg_records_dir', required=False, help='dir with tfrecords files containing png encoded bpg ims')
parser.add_argument('--batch_size', '-bsz', required=False, type=int, default=2)

# flags for cpm baseline
parser.add_argument('--cpm_log_dir', required=False, help='dir with cpm checkpoints')
parser.add_argument('--cpm_job_id', type=str, required=False, help='')

options = parser.parse_args()


def main(_opts):

    if _opts.compression == 'jpeg':
        assert _opts.records is not None, 'flag --records missing'

        from src.eval.hvs.eval_hvs_jpeg import EvalMSSSIMJpeg
        eval_hvs = EvalMSSSIMJpeg(_opts.dataset, _opts.records, _opts.batch_size)
        eval_hvs.run()

        return

    if _opts.compression == 'webp':
        assert _opts.records is not None, 'flag --records missing'

        from src.eval.hvs.eval_hvs_webp import EvalMSSSIMWebp
        eval_hvs = EvalMSSSIMWebp(_opts.dataset, _opts.records, _opts.batch_size)
        eval_hvs.run()

        return

    if _opts.compression == 'bpg':
        assert _opts.records is not None, 'flag --records missing'
        assert _opts.bpg_records_dir is not None, 'flag --bpg_records_dir missing'

        from src.eval.hvs.eval_hvs_bpg import EvalMSSSIMBpg
        eval_hvs = EvalMSSSIMBpg(_opts.dataset, _opts.bpg_records_dir, _opts.records, _opts.batch_size)
        eval_hvs.run()

        return

    if _opts.compression == 'rnn':
        assert _opts.rnn_ckpt_dir is not None, 'flag --rnn_ckpt_dir missing'
        assert _opts.records is not None, 'flag --records missing'
        from src.eval.hvs.eval_hvs_rnn import EvalMSSSIMRnn

        eval_hvs = EvalMSSSIMRnn(_opts.dataset, _opts.records, _opts.rnn_ckpt_dir, _opts.batch_size)
        eval_hvs.run()

        return

    if _opts.compression == 'cpm':
        assert _opts.cpm_log_dir is not None
        assert _opts.records is not None
        from src.eval.hvs.eval_hvs_cpm import EvalHVSCPM

        eval_accuracy = EvalHVSCPM(_opts.dataset, _opts.records, _opts.cpm_log_dir, _opts.cpm_job_id, _opts.batch_size)
        eval_accuracy.run()
        return

    raise ValueError('unknown compression method {}'.format(_opts.compression))


if __name__ == '__main__':
    main(options)
