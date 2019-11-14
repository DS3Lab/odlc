import argparse

parser = argparse.ArgumentParser()
# general
parser.add_argument('--dataset', required=True, help='either imagenet or stanford dogs')
parser.add_argument('--compression', required=False, help='one of bpg, jpeg, webp, rnn')
parser.add_argument('--records', required=False, help='records file with encoded image data')
parser.add_argument('--rnn_ckpt_dir', required=False, help='dir containing subfolder with checkpoints and config.json')
parser.add_argument(
    '--bpg_records_dir256', required=False,
    help='dir with tfrecords files containing png encoded bpg ims with resized images of resolution 256x256')
parser.add_argument(
    '--bpg_records_dir336', required=False,
    help='dir with tfrecords files containing png encoded bpg ims with resized images of resolution 336x336')

# flags for cpm baseline
parser.add_argument('--cpm_log_dir', required=False, help='dir with cpm checkpoints')
parser.add_argument('--cpm_job_id', type=str, required=False, help='')

# fine grained visual categorization
parser.add_argument('--fgvc_checkpoints', required=False, default=None, help='dir with model checkpoints')

options = parser.parse_args()


def main(_opts):
    if _opts.compression is None:
        assert _opts.records is not None
        from src.eval.accuracy.eval_accuracy_raw import EvalAccuracyRaw
        eval_accuracy = EvalAccuracyRaw(_opts.dataset, _opts.records, _opts.fgvc_checkpoints)
        eval_accuracy.run()
        return

    if _opts.compression == 'jpeg':
        assert _opts.records is not None
        from src.eval.accuracy.eval_accuracy_jpeg import EvalAccuracyJpeg
        eval_accuracy = EvalAccuracyJpeg(_opts.dataset, _opts.records, _opts.fgvc_checkpoints)
        eval_accuracy.run()
        return

    if _opts.compression == 'webp':
        assert _opts.records is not None
        from src.eval.accuracy.eval_accuracy_webp import EvalAccuracyWebp
        eval_accuracy = EvalAccuracyWebp(_opts.dataset, _opts.records, _opts.fgvc_checkpoints)
        eval_accuracy.run()
        return

    if _opts.compression == 'bpg':
        assert _opts.bpg_records_dir256 is not None
        assert _opts.bpg_records_dir336 is not None
        from src.eval.accuracy.eval_accuracy_bpg import EvalAccuracyBpg

        eval_accuracy = EvalAccuracyBpg(_opts.dataset, _opts.bpg_records_dir256, _opts.bpg_records_dir336,
                                        _opts.fgvc_checkpoints)
        eval_accuracy.run()
        return

    if _opts.compression == 'rnn':
        assert _opts.records is not None
        assert _opts.rnn_ckpt_dir is not None
        from src.eval.accuracy.eval_accuracy_rnn import EvalAccuracyRnn

        eval_accuracy = EvalAccuracyRnn(_opts.dataset, _opts.records, _opts.rnn_ckpt_dir, _opts.fgvc_checkpoints)
        eval_accuracy.run()
        return

    if _opts.compression == 'cpm':
        assert _opts.records is not None
        assert _opts.cpm_log_dir is not None
        from src.eval.accuracy.eval_accuracy_cpm import EvalAccuracyCPM

        eval_accuracy = EvalAccuracyCPM(_opts.dataset, _opts.records, _opts.cpm_log_dir, _opts.cpm_job_id,
                                        _opts.fgvc_checkpoints)
        eval_accuracy.run()
        return


if __name__ == '__main__':
    main(options)
